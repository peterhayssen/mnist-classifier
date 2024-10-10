import argparse
import torch
import os
import torch.optim as optim
import torch.nn as nn
from model import MLPNet, ConvNet 
from data_loader import get_data_loaders

def train_model(model, trainloader, testloader, epochs=5, lr=0.001):
    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()  # Set the model to training mode
        for images, labels in trainloader:
            # Move images and labels to the same device as the model
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Clear the gradients from the last step

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print statistics at the end of each epoch
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}")

    
    # After training, evaluate the model on the test set
    evaluate_model(model, testloader)

    # Save model after training
    save_model(model, model_name)


# Function to evaluate the model
def evaluate_model(model, testloader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate and print accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')


# Function to save the model to the models directory
def save_model(model, model_name):
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Print the directory and file path to check if they're correct
    print(f"Saving model to {model_dir}")
    
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")



if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train an MLP or CNN on the MNIST dataset.')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'],
                        help="Choose the model type: 'mlp' for MLPNet or 'cnn' for ConvNet (default: 'mlp')")
    parser.add_argument('--epochs', type=int, default=5,
                        help="Number of epochs to train the model (default: 5)")

    args = parser.parse_args()

    # Get data loaders
    trainloader, testloader = get_data_loaders(batch_size=64)

    # Set device (MPS if available, otherwise CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend for training.")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU instead.")

    # Choose the model based on the argument and set the model name
    if args.model == 'mlp':
        model = MLPNet()
        model_name = 'mlp_model'
    elif args.model == 'cnn':
        model = ConvNet()
        model_name = 'cnn_model'

    # Move the model to the appropriate device (MPS or CPU)
    model = model.to(device)

    # Train the model with the specified number of epochs
    train_model(model, trainloader, testloader, epochs=args.epochs, lr=0.001)