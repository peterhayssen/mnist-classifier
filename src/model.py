import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        # Input to hidden layer 1 (784 input features to 128 neurons)
        self.fc1 = nn.Linear(28 * 28, 128)
        # Hidden Layer 1 to hidden layer 2 (128 neurons to 64 neurons)
        self.fc2 = nn.Linear(128, 64)
        # Hidden Layer 2 to output layer (64 neurons to 10 output classes)
        self.fc3 = nn.Linear(64, 10)

    
    def forward(self, x):
        # Flatten the input (28x28 images to 784 element vectors)
        x = x.view(-1, 28 * 28)
        # Apply ReLU to first hidden layer
        x = F.relu(self.fc1(x))
        # Apply ReLU to second hidden layer
        x = F.relu(self.fc2(x))
        # Output layer (logits)
        x = self.fc3(x)
        return x
    
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer to reduce spatial dimensions (2x2 pooling)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layer: 64*7*7 input features (after pooling), 128 output features
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Fully connected output layer: 128 input features, 10 output features (for 10 digit classes)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Apply first convolutional layer and ReLU, then max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply second convolutional layer and ReLU, then max pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the output from convolutional layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Apply fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        # Output layer (logits)
        x = self.fc2(x)
        return x