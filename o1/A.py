import torch
import matplotlib.pyplot as plt
import pandas as pd
from components.LinearRegressionModel import LinearRegressionModel

# This is where our training data is stored
TRAINING_DATA_FILE_PATH = "o1/data/length_weight.csv"

# First, read the header from the file
headers = pd.read_csv(TRAINING_DATA_FILE_PATH, nrows=0).columns.tolist()
# Then read the data, using the header to key the columns
data = pd.read_csv(TRAINING_DATA_FILE_PATH, usecols=headers, sep=",")

# Convert each column of data to compatible tensors
data_x = torch.tensor(data["length"].values.tolist()).reshape(-1, 1)
data_y = torch.tensor(data["weight"].values.tolist()).reshape(-1, 1)

# Create an instance of a linear regression model
model = LinearRegressionModel()
# Use an Adam optimizer function instead of SGD.
optimizer = torch.optim.Adam([model.W, model.b], 0.1)

# Run through the epochs
for epoch in range(1000):
    model.loss(data_x, data_y).backward()
    optimizer.step()
    optimizer.zero_grad()

# Print the variables
print("W:    %f\nb:    %f\nloss: %f" %
      (model.W, model.b, model.loss(data_x, data_y)))

# Visualize result
plt.plot(data_x, data_y, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.tensor([[torch.min(data_x)], [torch.max(data_x)]])
plt.plot(x, model.predict(x).detach(), label='$f(x) = xW+b$')
plt.legend()
plt.show()
