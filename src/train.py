import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import XRayDataset
from src.model import get_pretrained_model
import matplotlib.pyplot as plt
import os

def train_model(data_dir='./data', num_epochs=15, batch_size=32, learning_rate=0.0001):
    """
    Trains the CNN model.
    Steps:
    1. Load data
    2. Set up the model, loss function, and optimizer
    3. Loop through the data for 'num_epochs' times
    4. Save the best model
    """
    
    # 1. Setup Device (Use GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 2. Load Data
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    train_dataset = XRayDataset(train_dir, is_training=True)
    val_dataset = XRayDataset(val_dir, is_training=False)
    
    if len(train_dataset) == 0:
        print("No training data found! Please add images to data/train/NORMAL and data/train/ABNORMAL")
        return
    
    print(f"Total images: {len(train_dataset)} (Train), {len(val_dataset)} (Val)")
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Setup Model, Loss, Optimizer
    model = get_pretrained_model(num_classes=2).to(device)
    # CrossEntropyLoss is standard for classification tasks
    criterion = nn.CrossEntropyLoss()
    # Adam is a versatile and fast optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # To store metrics for plotting
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    # 4. Training Loop
    for epoch in range(num_epochs):
        model.train() # Set to training mode (enables Dropout)
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad() # Clear previous gradients
            outputs = model(inputs) # Forward pass: Predict!
            loss = criterion(outputs, labels) # Calculate error
            loss.backward() # Backward pass: Calculate gradients
            optimizer.step() # Update weights to minimize error
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1) # Get the class with highest probability
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        
        # 5. Validation Loop
        model.eval() # Set to evaluation mode (disables Dropout for fair testing)
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad(): # Don't track gradients during testing (saves memory & speeds up)
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_loss /= len(val_loader)
        val_acc = correct_val / total_val
        
        # Save metrics for plotting later
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc: # Save only the best performing model
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Successfully saved best model! New best accuracy: {val_acc:.4f}")
            
    # 6. Plotting Results
    print("Training finished. Saving learning curves...")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    
    plt.savefig('training_curves.png')
    plt.close()
    
if __name__ == '__main__':
    train_model(data_dir='./data')
