import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import time
from mnist import load_test_data, load_training_data

# Define a simple neural network model


class MNISTModel(torch.nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # First convolution layer
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Second convolution layer
        self.conv2 = torch.nn.Conv2d(
            32, 64, kernel_size=3, stride=1, padding=1)
        # Fully connected layer 1
        self.fc1 = torch.nn.Linear(64 * 28 * 28, 128)
        # Output layer (10 classes for digits 0-9)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Function to train the model
def train(model, criterion, optimizer, num_epochs=5):
    train_loader = DataLoader(load_training_data(square_images=True),
                              batch_size=64, shuffle=True)

    print("Training the model... ")
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        print(
            f"    Epoch [{epoch+1}/{num_epochs}] running... ", end="", flush=True)
        start_time = time.time()

        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()  # Zero out gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update weights
            running_loss += loss.item()

        print(
            f"DONE({time.time() - start_time:.1f}s) Loss: {running_loss/len(train_loader):.5f}")
    print("DONE")

# Function to test the model


def test(model):
    test_loader = DataLoader(load_test_data(), batch_size=1000, shuffle=False)

    print("Testing the model... ")
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"    Test Accuracy: {100 * correct / total:.2f}%")
    print("DONE")


def create_model():
    # Initialize the model, loss function, and optimizer
    model = MNISTModel()
    # Cross-entropy loss for multi-class classification
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Train and test the model
    train(model, criterion, optimizer, num_epochs=5)
    test(model)

    # Save the model (optional)
    print("Saving model... ", end="", flush=True)
    torch.save(model.state_dict(), "mnist_model.pth")
    print("DONE")


def test_saved_model():
    model = MNISTModel()
    # Load the model
    model.load_state_dict(torch.load("mnist_model.pth", weights_only=True))
    # Test the model
    test(model)


create_model()
