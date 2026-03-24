import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import os



# Set page config
st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTitle {
        text-align: center;
        color: #667eea;
    }
    </style>
    """, unsafe_allow_html=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model import BoneCNN

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = 'bone_fraction.pth'

    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Make sure it is included in the repository.")
        st.stop()
        
    model = BoneCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

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

def predict(image):
    """Make prediction on image"""
    model = load_model()
    img_tensor = preprocess_image(image)
    
    with torch.no_grad():
        logits = model(img_tensor)  # [1, 2]
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item() * 100
    
    class_names = ['fractured', 'notfractured']
    
    return {
        'prediction': class_names[pred_idx],
        'confidence': confidence,
        'notfractured_prob': probs[0, 0].item() * 100,
        'fractured_prob': probs[0, 1].item() * 100
    }

# Main App
st.markdown("# 🏥 Bone Fracture Detection")
st.markdown("Upload X-ray images to detect bone fractures using AI")
st.divider()

# Sidebar info
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown("""
    This application uses a trained deep learning model to detect bone fractures in X-ray images.
    
    **Model Details:**
    - Framework: PyTorch
    - Input: 150×150 RGB images
    - Output: Binary classification
    - Processing: GPU/CPU automatic detection
    """)
    
    st.markdown("---")
    st.markdown("**Device:** " + ("GPU (CUDA)" if torch.cuda.is_available() else "CPU"))

# File uploader
uploaded_file = st.file_uploader("Upload an X-ray image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📷 Uploaded Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.markdown("### 🔍 Processing...")
        with st.spinner("Analyzing image..."):
            results = predict(image)
        
        st.markdown("### ✅ Results")
        
        # Display prediction
        prediction = results['prediction']
        confidence = results['confidence']
        
        if prediction == 'fractured':
            st.error(f"### Fractured\n**Confidence: {confidence:.2f}%**")
        else:
            st.success(f"### Not Fractured\n**Confidence: {confidence:.2f}%**")
    
    st.divider()
    
    # Detailed results
    st.markdown("### 📊 Detailed Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Confidence", f"{confidence:.2f}%")
    
    with col2:
        st.metric("No Fracture", f"{results['notfractured_prob']:.2f}%")
    
    with col3:
        st.metric("Fracture", f"{results['fractured_prob']:.2f}%")
    
    # Probability chart
    st.markdown("### 📈 Probability Distribution")
    prob_data = {
        'No Fracture': results['notfractured_prob'],
        'Fracture': results['fractured_prob']
    }
    st.bar_chart(prob_data)

else:
    st.info("👆 Upload an X-ray image to get started!")
