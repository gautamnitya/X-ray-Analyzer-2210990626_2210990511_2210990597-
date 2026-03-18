from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
import base64
import torch
import os
from src.inference import load_model_for_inference, predict_and_explain

app = FastAPI(title="AI X-Ray Analysis")

# Setup templates directory
os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")

# Try to load model globally when server starts
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading Model on {device}...")
model = load_model_for_inference('best_model.pth', device)

def image_to_base64(img_array):
    """Converts a numpy image back to a base64 string for HTML display."""
    _, buffer = cv2.imencode('.jpg', img_array)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Renders the main upload page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_xray(request: Request, file: UploadFile = File(...)):
    """Receives the image, runs the AI pipeline, and returns results to the UI."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
    
    try:
        # Run our AI pipeline
        prediction, confidence, enhanced_img, gradcam_img = predict_and_explain(img, model, device)
        
        # Convert images to base64 so we can show them on the webpage without saving to disk
        enhanced_b64 = image_to_base64(enhanced_img)
        # OpenCV uses BGR natively, but the model outputs RGB for Grad-CAM. Let's make sure it shows correct format.
        gradcam_b64 = image_to_base64(cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%",
            "enhanced_image": enhanced_b64,
            "gradcam_image": gradcam_b64,
            "success": True
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e),
            "success": False
        })

if __name__ == "__main__":
    print("Starting server... Access at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
