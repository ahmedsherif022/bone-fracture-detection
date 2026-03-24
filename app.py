import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import io

try:
    from torchvision import transforms
except RuntimeError as e:
    # Fallback - define transforms manually if torchvision has issues
    transforms = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model import BoneCNN

# Load Model
model_path = os.path.join('saved_models', 'bone_fraction.pth')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Make sure it is included in the repository.")

model = BoneCNN().to(device)
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Image preprocessing
img_size = 256

def preprocess_image(image):
    """Convert PIL image to tensor with proper preprocessing - matching training pipeline"""
    img = image.convert('RGB')
    img = img.resize((img_size, img_size))
    # Convert to grayscale then back to 3 channels (matching training pipeline)
    img_gray = img.convert('L')
    img_array = np.array(img_gray) / 255.0  # Shape: (H, W)
    # Replicate grayscale values across 3 channels
    img_array_rgb = np.stack([img_array, img_array, img_array], axis=2)  # Shape: (H, W, 3)
    img_tensor = torch.from_numpy(img_array_rgb).permute(2, 0, 1).float()
    # Normalize using the same values as training: (0.5, 0.5, 0.5)
    img_tensor = (img_tensor - 0.5) / 0.5
    return img_tensor.unsqueeze(0).to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and preprocess image
        image = Image.open(file.stream)
        img_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            logits = model(img_tensor)  # [1, 2]
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item() * 100
        
        # Class names (must match dataset order: fractured=0, notfractured=1)
        class_names = ['fractured', 'notfractured']
        predicted_class = class_names[pred_idx]
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': f'{confidence:.2f}%',
            'notfractured_prob': f'{probs[0, 0].item()*100:.2f}%',
            'fractured_prob': f'{probs[0, 1].item()*100:.2f}%'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
