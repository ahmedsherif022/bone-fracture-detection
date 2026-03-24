## Quick Start Guide

I've created a professional web application for your bone fracture detection model. You have **two options**:

### Option 1: Flask Application (Recommended for Production)

#### Setup:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
```

Then open your browser to: **http://localhost:5000**

**Features:**
- Custom-designed professional UI
- Drag-and-drop image upload
- Real-time predictions with confidence scores
- Shows probability for both classes
- Responsive design (works on mobile too)

---

### Option 2: Streamlit Application (Easiest Setup)

#### Setup:
```bash
# Install Streamlit
pip install streamlit

# Run the Streamlit app
streamlit run app_streamlit.py
```

**Features:**
- Simpler to set up
- Built-in charts and metrics
- Minimal configuration needed
- Great for quick prototyping

---

## Files Created:

1. **app.py** - Flask web server with beautiful UI
2. **app_streamlit.py** - Streamlit alternative (simpler)
3. **templates/index.html** - Web interface for Flask app
4. **requirements.txt** - All dependencies
5. **README.md** - Full documentation

---

## How to Use:

1. Navigate to your project folder:
   ```bash
   cd "e:\learning CNN"
   ```

2. Install dependencies (first time only):
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   # Flask version:
   python app.py
   
   # OR Streamlit version:
   streamlit run app_streamlit.py
   ```

4. Open your browser and upload an X-ray image

---

## Model Configuration:

The app automatically:
✅ Loads `bone_fraction.pth` from `saved_models/` folder
✅ Resizes images to 150×150 pixels
✅ Normalizes pixel values
✅ Uses GPU if available, CPU otherwise
✅ Shows prediction confidence and probabilities

---

## Troubleshooting:

**"Module not found" error:**
```bash
pip install -r requirements.txt
```

**Port 5000 in use:**
Edit `app.py` and change `port=5000` to `port=8000`

**CUDA/GPU errors:**
The app works on CPU too - it'll just be slower

---

## Next Steps:

- For **production deployment**: Use Flask with Gunicorn
- For **quick testing**: Use Streamlit
- To **customize**: Edit the HTML in `templates/index.html`

Let me know if you need any modifications!
