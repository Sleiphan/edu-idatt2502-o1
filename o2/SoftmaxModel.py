import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mnist import load_test_data, load_training_data
import matplotlib.pyplot as plt
import numpy as np

train_loader = DataLoader(load_training_data(), batch_size=64, shuffle=True)
test_loader = DataLoader(load_test_data(), batch_size=1000, shuffle=False)


class SoftmaxModel:
    def __init__(self):
        self.W = torch.ones([784, 10], requires_grad=True)
        self.b = torch.ones([1, 10], requires_grad=True)

    # Predictor
    def f(self, x):
        return torch.nn.functional.softmax(x @ self.W + self.b, dim=1)

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


# Initialize model, loss function, and optimizer
model = SoftmaxModel()
# criterion = nn.CrossEntropyLoss()  # Combines softmax and cross-entropy loss

optimizer = optim.Adam([model.W, model.b], lr=0.001)

# Training loop
for epoch in range(5):
    for batch in train_loader:
        labels, images = batch
        outputs = model.f(images)
        model.loss(outputs, labels).backward()
        optimizer.step()
        optimizer.zero_grad()

# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

# great_images = model.coefficients.detach().reshape(10, 28, 28).numpy()

# fig = plt.figure('Photos')
# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(great_images[i])
#     plt.title(f'W: {i}')
#     plt.xticks([])
#     plt.yticks([])

# plt.show()
