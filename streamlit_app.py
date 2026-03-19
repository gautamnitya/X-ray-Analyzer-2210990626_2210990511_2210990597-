import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from src.inference import load_model_for_inference, predict_and_explain
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI X-Ray Analysis System",
    page_icon="🏥",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #3498db; color: white; }
    [data-testid="stMetricValue"] { font-size: 28px; }
    .stMetric { 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR & MODEL LOADING ---
st.sidebar.title("🩺 Control Panel")
st.sidebar.info("This system analyzes Chest X-Ray images using a ResNet18 Deep Learning model with Grad-CAM explainability.")

@st.cache_resource
def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_for_inference('best_model.pth', device)
    return model, device

model, device = get_model()

# --- HEADER ---
st.title("🏥 AI-Driven X-Ray Diagnostic System")
st.write("---")

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("📂 Upload Center")
    uploaded_file = st.file_uploader("Drop an X-Ray image here...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Original Image", use_container_width=True)
        
        analyze_btn = st.button("🚀 Run AI Analysis")

with col2:
    st.header("🔍 Analytical Results")
    
    if uploaded_file is not None and analyze_btn:
        with st.spinner("Processing image and generating explainability heatmap..."):
            # Convert PIL to format for OpenCV/PyTorch
            img_array = np.array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) # Convert to BGR for our pipeline
            
            try:
                # 1. Run inference
                prediction, confidence, enhanced_img, gradcam_img = predict_and_explain(img_array, model, device)
                
                # 2. Display Diagnosis Metrics
                st.write("### AI Prediction")
                m1, m2 = st.columns(2)
                
                color = "green" if prediction == "NORMAL" else "red"
                m1.metric("Diagnosis", prediction, delta=None)
                m2.metric("Confidence Score", f"{confidence:.1f}%")
                
                if prediction == "ABNORMAL":
                    st.error(f"⚠️ **Result: {prediction}** — Possible Pneumonia or medical abnormality detected.")
                else:
                    st.success(f"✅ **Result: {prediction}** — The X-ray appears clear/normal.")
                
                st.write("---")
                
                # 3. Enhanced Visualization
                v_col1, v_col2 = st.columns(2)
                
                with v_col1:
                    st.subheader("📸 Enhanced Clarity")
                    st.caption("Noise reduction & contrast adjustment applied.")
                    st.image(enhanced_img, use_container_width=True)
                    
                with v_col2:
                    st.subheader("🔥 AI Attention")
                    st.caption("Grad-CAM highlights the critical features.")
                    st.image(gradcam_img, use_container_width=True)
                    
                st.info("**Explainability Note:** The heatmap highlights areas that contributed most to the AI's diagnosis. This helps clinicians verify if the model is focusing on the correct anatomical regions.")
                
            except Exception as e:
                st.error(f"Inference error: {e}")
    else:
        st.info("Waiting for image upload and analysis trigger...")

# --- FOOTER ---
st.write("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 14px;'>
    Developed for Research & Education Purposes. <br>
    <b>Model:</b> ResNet18 Transfer Learning | <b>XAI:</b> Grad-CAM | <b>Backend:</b> PyTorch
</div>
""", unsafe_allow_html=True)
