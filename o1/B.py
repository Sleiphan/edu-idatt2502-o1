import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from components.LinearRegressionModel import MultipleLinearRegression

# This is where our training data is stored
TRAINING_DATA_FILE_PATH = "o1/data/day_length_weight.csv"
headers = pd.read_csv(TRAINING_DATA_FILE_PATH, nrows=0).columns.tolist()
data = pd.read_csv(TRAINING_DATA_FILE_PATH, usecols=headers, sep=",")

# Convert each column of data to compatible tensors
data_X = torch.tensor(list(zip(data["length"], data["weight"]))).reshape(-1, 2)
data_Y = torch.tensor(data["day"].values.tolist()).reshape(-1, 1)

model = MultipleLinearRegression(2, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

# Run through the epochs
for epoch in range(1000):
    predictions = model(data_X)
    loss = criterion(predictions, data_Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model)

fig = plt.figure()
# 3D plots require a subplot with 3D projection enabled
ax = fig.add_subplot(projection='3d')

ax.plot(data["length"].values.tolist(),
        data["weight"].values.tolist(),
        data["day"].values.tolist(),
        'o')

# Create the vector [0., 0.01, ..., 0.99]
x = torch.arange(start=0., end=1., step=0.2)
# Create the square matrix [[0., 0.01, ..., 0.99], ..., [0., 0.01, ..., 0.99]]
x = x.expand(x.shape[0], -1)
# Transpose of x: [[0., ..., 0.], [0.01, ..., 0.01], ... [0.99, ..., 0.99]]
y = x.T
# * and + are element-wise operators resulting in a matrix with the same shape as x and y

# Scale coordinate matrices according to input data
x = x * (data["length"].max() - data["length"].min())
y = y * (data["weight"].max() - data["weight"].min())
x += x + data["length"].min()
y += y + data["weight"].min()

# Create an tensor containing all unique coordinates we want to cover
coords = torch.stack((x.flatten(), y.flatten()), dim=1)
z = model(coords).reshape(x.shape[0], x.shape[1])

# Convert the tensors to numpy arrays for plotting
x_np = x.detach().numpy()
y_np = y.detach().numpy()
z_np = z.detach().numpy()  # Use detach to remove gradients before converting to numpy

ax.plot_wireframe(
    x_np, y_np, z_np,
    label="z = $B_0$ + $B_1$ * x + $B_2$ * y")

ax.set_xlabel('Length')
ax.set_ylabel('Weight')
ax.set_zlabel('Day')
plt.legend()
plt.show()
