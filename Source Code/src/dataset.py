import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from src.enhance import preprocess_xray  # ensure python path handles src package

class XRayDataset(Dataset):
    """
    A custom PyTorch Dataset for our X-Ray images.
    It reads images from a folder, applies our OpenCV enhancements, 
    and then applies PyTorch augmentations (like random rotation) to help the model generalize.
    """
    def __init__(self, data_dir, is_training=True):
        self.data_dir = data_dir
        self.is_training = is_training
        
        # We assume the directory has two subfolders: "NORMAL" and "ABNORMAL" (or PNEUMONIA)
        self.classes = ['NORMAL', 'ABNORMAL'] 
        self.image_paths = []
        self.labels = []
        
        # Scan directory and record file paths and their labels (0 or 1)
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory {class_dir} not found. Ensure dataset is placed correctly.")
                continue
                
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(label_idx)
                    
        # PyTorch Data Augmentation
        # Only apply random changes during training, not during testing/validation!
        if self.is_training:
            self.pytorch_transforms = transforms.Compose([
                transforms.ToPILImage(), # Convert numpy array to PIL Image for PyTorch transforms
                transforms.RandomRotation(15), # Randomly rotate by up to 15 degrees
                transforms.RandomHorizontalFlip(), # Randomly flip left-right
                transforms.ToTensor() # Convert back to PyTorch Tensor with shape (C, H, W)
            ])
        else:
            self.pytorch_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 1. Apply our custom OpenCV enhancement (returns numpy array 0-1)
        # We discard the first return value (display image) and keep the second (normalized float array)
        _, enhanced_array = preprocess_xray(img_path)
        
        # 2. To use the ToPILImage transform properly, we first scale it back to 0-255 uint8, 
        # then the ToTensor transform will scale it back down to 0-1 for the model 
        # (This is a quirk of mixing OpenCV and PyTorch transforms)
        enhanced_array_uint8 = (enhanced_array * 255).astype('uint8')
        
        # 3. Apply PyTorch augmentations
        img_tensor = self.pytorch_transforms(enhanced_array_uint8)
        
        return img_tensor, torch.tensor(label, dtype=torch.long)
