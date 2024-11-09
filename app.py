# app.py
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
import openai
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import io
import os
import base64
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Replace with a fixed secret key in production
CORS(app)

# Ensure 'static/annotated' directory exists
ANNOTATED_DIR = os.path.join('static', 'annotated')
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# Set maximum upload size to 16 MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize ONNX Runtime session
ort_session = ort.InferenceSession("models/best.onnx")

# Define the input and output names
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# Define class names (update these based on your dataset)
class_names = ["blackheads", "dark spot", "nodules", "papules", "pustules", "whiteheads"]

def preprocess_image(image: Image.Image, img_size: int = 640):
    """
    Preprocess the input image for YOLOv5 ONNX model.
    """
    # Resize and pad image to img_size x img_size
    img = image.resize((img_size, img_size))
    img = np.array(img).astype(np.float32)
    img /= 255.0  # Normalize to [0,1]
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.ascontiguousarray(img)
    return img

def postprocess(outputs, conf_threshold=0.25, iou_threshold=0.45):
    """
    Postprocess the outputs from the YOLOv5 ONNX model.
    Applies confidence thresholding and non-maximum suppression.
    """
    detections = []
    preds = outputs[0]  # preds shape: [1, 25200, 11]

    # Remove batch dimension
    preds = np.squeeze(preds, axis=0)  # [25200, 11]

    # Convert to NumPy array if not already
    if isinstance(preds, list):
        preds = np.array(preds)

    # Extract bbox coordinates, confidence, and class probabilities
    bbox = preds[:, :4]  # [x1, y1, x2, y2]
    confidence = preds[:, 4]  # [confidence score]
    class_probs = preds[:, 5:]  # [6 class probabilities]

    # Get class IDs with highest probability
    class_ids = np.argmax(class_probs, axis=1)
    class_confidences = class_probs[np.arange(len(class_probs)), class_ids]

    # Multiply object confidence with class confidence
    scores = confidence * class_confidences

    # Filter out low confidence detections
    mask = scores >= conf_threshold
    bbox = bbox[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(bbox) == 0:
        return detections

    # Convert to list of bounding boxes and scores
    boxes = bbox.tolist()
    scores = scores.tolist()

    # Apply Non-Max Suppression using OpenCV
    boxes_cv = []
    for box in boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        boxes_cv.append([x1, y1, width, height])

    indices = cv2.dnn.NMSBoxes(
        boxes_cv,
        scores,
        conf_threshold,
        iou_threshold
    )

    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, x2, y2 = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            class_name = class_names[class_id] if class_id < len(class_names) else "unknown"

            detections.append({
                "xmin": float(x1),
                "ymin": float(y1),
                "xmax": float(x2),
                "ymax": float(y2),
                "confidence": float(score),
                "class": int(class_id),
                "name": class_name
            })

    return detections

def annotate_image(image: Image.Image, detections):
    """
    Draw bounding boxes and labels on the image based on detections.
    Returns the annotated PIL Image.
    """
    img = np.array(image).copy()

    for det in detections:
        x1 = int(det['xmin'])
        y1 = int(det['ymin'])
        x2 = int(det['xmax'])
        y2 = int(det['ymax'])
        confidence = det['confidence']
        class_name = det['name']

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put label
        label = f"{class_name}: {confidence:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    annotated_image = Image.fromarray(img)
    return annotated_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        return render_template('result.html', detections=[], annotated_image=None, error="Unsupported file type.")

    try:
        # Read image in PIL format
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Preprocess the image
        input_tensor = preprocess_image(image)

        # Run inference
        outputs = ort_session.run([output_name], {input_name: input_tensor})  # List of outputs

        # Postprocess the outputs
        detections = postprocess(outputs)

        # Reset chat history and add detection results as system prompt
        session['chat_history'] = []
        if detections:
            detection_summary = "Detected the following issues in the uploaded image:\n"
            for det in detections:
                detection_summary += f"- {det['name']} with confidence {det['confidence']:.2f}\n"
            session['chat_history'].append({'role': 'system', 'content': detection_summary})
        else:
            session['chat_history'].append({'role': 'system', 'content': "No acne-related issues were detected in the uploaded image."})

        # Annotate image
        annotated_image = annotate_image(image, detections)

        # Save annotated image to 'static/annotated' with unique name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        annotated_image_filename = f"{timestamp}_annotated.jpg"
        annotated_image_path = os.path.join(ANNOTATED_DIR, annotated_image_filename)
        annotated_image.save(annotated_image_path)

        # Create the URL for the annotated image
        annotated_image_url = url_for('static', filename=f"annotated/{annotated_image_filename}")

        # Render the results page with chat
        return render_template(
            'result.html',
            detections=detections,
            annotated_image=annotated_image_url,
            error=None,
            chat_history=session['chat_history']
        )

    except Exception as e:
        print("Error during detection:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/chat/message', methods=['POST'])
def chat_message():
    user_message = request.form.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided.'}), 400

    # Retrieve chat history from session
    chat_history = session.get('chat_history', [])

    # Append user's message to chat history
    chat_history.append({'role': 'user', 'content': user_message})

    # Prepare messages for OpenAI
    messages = [
        {"role": "system", "content": "You are a professional dermatologist. Provide accurate and helpful advice based on the user's acne detection results."}
    ]
    messages.extend(chat_history)

    try:
        # Call OpenAI's ChatGPT
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )

        assistant_message = response.choices[0].message['content']

        # Append assistant's message to chat history
        chat_history.append({'role': 'assistant', 'content': assistant_message})

        # Update session chat history
        session['chat_history'] = chat_history

        return jsonify({'message': assistant_message})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
