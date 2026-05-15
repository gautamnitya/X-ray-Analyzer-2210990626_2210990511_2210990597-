import torch
import numpy as np
from torch.utils.data import DataLoader
from src.dataset import XRayDataset
from src.model import get_pretrained_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(data_dir='./data', model_path='best_model.pth', batch_size=32):
    """
    Evaluates the trained model on test data, displaying Accuracy, Precision, Recall, and F1.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    test_dir = os.path.join(data_dir, 'test')
    
    test_dataset = XRayDataset(test_dir, is_training=False)
    if len(test_dataset) == 0:
        print("No test data found. Please add images to data/test/NORMAL and data/test/ABNORMAL")
        return
        
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = get_pretrained_model(num_classes=2).to(device)
    
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found! Please train the model first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running predictions...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate metrics
    print("\nClassification Report (Precision, Recall, F1-Score, Accuracy):")
    target_names = test_dataset.classes
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Saved confusion matrix to confusion_matrix.png")

if __name__ == '__main__':
    evaluate_model(data_dir='./data')
