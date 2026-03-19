import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from src.model import get_pretrained_model
from src.dataset import XRayDataset
from src.enhance import preprocess_xray
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
import seaborn as sns
from torch.utils.data import DataLoader

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def generate_preprocessing_viz(image_path, save_path='paper_preprocessing.png'):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    
    # 1. Filtered
    img_filtered = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_contrast = clahe.apply(img_filtered)
    
    # 3. Sharpened
    gaussian_blur = cv2.GaussianBlur(img_contrast, (9, 9), 10.0)
    img_sharpened = cv2.addWeighted(img_contrast, 1.5, gaussian_blur, -0.5, 0)
    
    titles = ['Original', 'Gaussian Filter', 'CLAHE Enhancement', 'Edge Enhancement']
    images = [img, img_filtered, img_contrast, img_sharpened]
    
    plt.figure(figsize=(15, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved preprocessing visualization to {save_path}")

def generate_metrics_viz(model, test_loader, device, save_prefix='paper_metrics'):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1]) # Probability of ABNORMAL
            all_preds.extend(preds.cpu().numpy())
            
    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{save_prefix}_cm.png', dpi=300)
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_prefix}_roc.png', dpi=300)
    plt.close()
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(f'{save_prefix}_pr.png', dpi=300)
    plt.close()
    
    print(f"Saved metric visualizations to {save_prefix}_*.png")

def generate_gradcam_viz(model, device, data_dir, save_path='paper_gradcam.png'):
    # Get one Normal and one Abnormal sample
    normal_dir = os.path.join(data_dir, 'test', 'NORMAL')
    abnormal_dir = os.path.join(data_dir, 'test', 'ABNORMAL')
    
    normal_img_path = os.path.join(normal_dir, os.listdir(normal_dir)[0])
    abnormal_img_path = os.path.join(abnormal_dir, os.listdir(abnormal_dir)[0])
    
    target_layer = model.layer4[-1] # Last conv layer of ResNet18
    grad_cam = GradCAM(model, target_layer)
    
    samples = [
        ('Normal', normal_img_path),
        ('Abnormal', abnormal_img_path)
    ]
    
    plt.figure(figsize=(10, 8))
    
    for i, (label, img_path) in enumerate(samples):
        # Preprocess
        display_img, normalized_img = preprocess_xray(img_path)
        input_tensor = torch.from_numpy(normalized_img).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Generate CAM
        cam = grad_cam.generate(input_tensor)
        
        # Overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlayed = cv2.addWeighted(cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB), 0.6, heatmap, 0.4, 0)
        
        plt.subplot(2, 2, i*2 + 1)
        plt.imshow(display_img, cmap='gray')
        plt.title(f'Original ({label})')
        plt.axis('off')
        
        plt.subplot(2, 2, i*2 + 2)
        plt.imshow(overlayed)
        plt.title(f'Grad-CAM ({label})')
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Grad-CAM visualization to {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = './data'
    model_path = 'best_model.pth'
    
    if not os.path.exists(model_path):
        print("Model not found. Please train it first.")
        return
        
    model = get_pretrained_model(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # 1. Preprocessing Visualization
    test_normal = os.path.join(data_dir, 'test', 'NORMAL')
    if os.path.isdir(test_normal) and os.listdir(test_normal):
        sample_path = os.path.join(test_normal, os.listdir(test_normal)[0])
        generate_preprocessing_viz(sample_path)
    
    # 2. Metrics Visualization
    test_dataset = XRayDataset(os.path.join(data_dir, 'test'), is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    generate_metrics_viz(model, test_loader, device)
    
    # 3. Grad-CAM Visualization
    generate_gradcam_viz(model, device, data_dir)

if __name__ == '__main__':
    main()
