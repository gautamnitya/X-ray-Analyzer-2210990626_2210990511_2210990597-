import cv2
import numpy as np

def preprocess_xray(image_path_or_array, target_size=(224, 224)):
    """
    Applies the image enhancement techniques described in the research:
    1. Filtering (Noise reduction)
    2. Contrast adjustment
    3. Edge enhancement
    4. Normalization
    
    Args:
        image_path_or_array: Path to image string or raw numpy array
        target_size: Resize dimensions for the CNN
        
    Returns:
        img_display: The enhanced image ready to be shown to the user (0-255 uint8)
        img_tensor: The normalized RGB image ready for CNN input (0-1 float32)
    """
    
    # Check if input is a path (string) or an image array
    if isinstance(image_path_or_array, str):
        # 1. Read the image in grayscale
        img = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image at {image_path_or_array}")
    else:
        # If it's already an array (e.g., from web upload)
        # Ensure it's grayscale
        if len(image_path_or_array.shape) > 2:
            img = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2GRAY)
        else:
            img = image_path_or_array

    # Resize image to target size for CNN
    img = cv2.resize(img, target_size)
    
    # --- 1. FILTERING ---
    # Reduce noise using Gaussian Blur. This smoothens the image slightly.
    img_filtered = cv2.GaussianBlur(img, (5, 5), 0)
    
    # --- 2. CONTRAST ADJUSTMENT ---
    # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This brings out dark details in the X-ray without overexposing bright parts
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_contrast = clahe.apply(img_filtered)
    
    # --- 3. EDGE ENHANCEMENT ---
    # Unsharp Masking technique: Subtract a blurred version from the original
    # This sharpens the edges, making bones and lung borders more distinct
    gaussian_blur = cv2.GaussianBlur(img_contrast, (9, 9), 10.0)
    img_sharpened = cv2.addWeighted(img_contrast, 1.5, gaussian_blur, -0.5, 0)
    
    # --- 4. NORMALIZATION ---
    # Convert single channel grayscale to 3 channels by replicating (useful for standard CNNs)
    # Even though X-rays are 1 channel, pre-trained CNNs (like ResNet) expect 3 channels (RGB)
    img_rgb = cv2.cvtColor(img_sharpened, cv2.COLOR_GRAY2RGB)
    
    # Scale pixel values down to [0, 1] range as a float32 array 
    # This is standard practice before feeding data into Neural Networks
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # We return the sharpened 8-bit image for UI display, and the float dataset for the ML model
    return img_sharpened, img_normalized

if __name__ == "__main__":
    print("This module provides X-Ray enhancement functions: Filtering, Contrast, Edge Enhancement, Normalization.")
