import torch
import torch.nn as nn
import torchvision.models as models

class CustomXRayCNN(nn.Module):
    """
    A standalone simple Convolutional Neural Network (CNN).
    This implements learning from scratch. It is highly readable and 
    great for beginners to learn exactly how PyTorch layers work.
    """
    def __init__(self, num_classes=2): # 2 categories: Normal vs Abnormal
        super(CustomXRayCNN, self).__init__()
        
        # Convolution Layer 1 -> Extracts basic features (edges, corners)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU() # Activation function
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrinks image by half
        
        # Convolution Layer 2 -> Extracts more complex patterns
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolution Layer 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected (Dense) Layers -> Makes the final decision
        # After 3 MaxPool operations on a 224x224 image, size is 28x28
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.relu4 = nn.ReLU()
        
        # Dropout randomly turns off 50% of neurons to prevent 'overfitting' (memorizing data)
        self.dropout = nn.Dropout(0.5) 
        
        # Final layer outputs the 2 classes
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Pass image through CNN layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        # Flatten the 3D tensor arrays into a 1D vector
        x = x.view(x.size(0), -1) 
        
        # Pass through Dense layers
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_pretrained_model(num_classes=2):
    """
    Uses Transfer Learning with a pre-trained CNN (ResNet18).
    Because it was already trained on millions of images, it will learn 
    X-ray features much faster and achieve higher accuracy (>95% usually).
    """
    # Load ResNet18 model architecture and pre-trained weights
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # The original ResNet has 1000 output classes. 
    # We replace the final fully-connected (fc) layer to output just our 2 classes.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
