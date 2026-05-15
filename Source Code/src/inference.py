import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.model import get_pretrained_model
from src.enhance import preprocess_xray
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

def load_model_for_inference(model_path='best_model.pth', device='cpu'):
    """
    Loads our trained model for making predictions.
    """
    model = get_pretrained_model(num_classes=2).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print(f"Warning: {model_path} not found. Using untrained model for demonstration.")
        
    model.eval()
    return model

def predict_and_explain(image_path_or_array, model, device='cpu'):
    """
    1. Enhances the input image
    2. Makes a prediction
    3. Generates a Grad-CAM heatmap to see what the model is looking at
    """
    # 1. Preprocess
    img_display, img_normalized = preprocess_xray(image_path_or_array)
    
    # Needs to be shape (1, C, H, W) for PyTorch
    # Our normalized image is (H, W, C), so we permute
    tensor_img = torch.tensor(img_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 2. Predict
    with torch.no_grad():
        outputs = model(tensor_img)
        # Apply Softmax to get probabilities (between 0 and 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, 0)
        
    class_names = ['NORMAL', 'ABNORMAL']
    prediction = class_names[predicted_class.item()]
    conf_score = float(confidence.item()) * 100
    
    # 3. Grad-CAM (Explainability)
    # We tell Grad-CAM which layer to look at. For ResNet18, layer4 is the final block of CNN layers.
    target_layers = [model.layer4[-1]]
    
    # Initialize the Grad-CAM object
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Generate the heatmap.
    grayscale_cam = cam(input_tensor=tensor_img, targets=None)[0, :]
    
    # Overlay the heatmap on the enhanced image
    # cv2 color expectation for show_cam_on_image is float32 RGB between 0 and 1
    rgb_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    visualization = show_cam_on_image(rgb_display, grayscale_cam, use_rgb=True)
    
    # Convert back to 0-255 uint8 for easy saving/displaying
    visualization = (visualization * 255).astype(np.uint8)
    
    return prediction, conf_score, img_display, visualization

if __name__ == "__main__":
    print("Inference module loaded. Use predict_and_explain() in the web app.")
