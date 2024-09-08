import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

TRAINING_DATA_FILE_PATH = "o2/data/mnist_train.csv"
TEST_DATA_FILE_PATH = "o2/data/mnist_test.csv"

print("Loading training data... ", end="")
mnist_data_training = pd.read_csv(
    TRAINING_DATA_FILE_PATH, sep=",", header=None)
print("DONE")

print("Loading test data... ", end="")
mnist_data_test = pd.read_csv(TEST_DATA_FILE_PATH, sep=",", header=None)
print("DONE")

print("Setup data structures... ", end="")
training_labels = torch.tensor(
    mnist_data_training[0].to_numpy())
training_images = torch.tensor(
    mnist_data_training.loc[:, mnist_data_training.columns != 0].to_numpy(), dtype=torch.float32)

test_labels = torch.tensor(mnist_data_test[0].to_numpy())
test_images = torch.tensor(
    mnist_data_test.loc[:, mnist_data_test.columns != 0].to_numpy(), dtype=torch.float32)

# Reshape the images to be 28x28
training_images = training_images.reshape(-1, 1, 28, 28)
test_images = test_images.reshape(-1, 1, 28, 28)

# Create DataLoaders for batching
train_dataset = TensorDataset(training_images, training_labels)
test_dataset = TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("DONE")

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


# Initialize the model, loss function, and optimizer
model = MNISTModel()
# Cross-entropy loss for multi-class classification
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Function to train the model


def train(model, train_loader, criterion, optimizer, num_epochs=5):
    print("Training model... ")
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()  # Zero out gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update weights
            running_loss += loss.item()
        print(
            f"    Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    print("DONE")

# Function to test the model


def test(model, test_loader):
    print("Testing model... ")
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


# Train and test the model
train(model, train_loader, criterion, optimizer, num_epochs=5)
test(model, test_loader)

# Save the model (optional)
print("Saving model... ", end="")
torch.save(model.state_dict(), "mnist_model.pth")
print("DONE")
# model = torch.load("mnist_model.pth")
