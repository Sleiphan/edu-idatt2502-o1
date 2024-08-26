import torch
import matplotlib.pyplot as plt
import pandas as pd
from components.LinearRegressionModel import LinearRegressionModel

TRAINING_DATA_FILE_PATH = "o1/data/length_weight.csv"

# First, read the header from the file
headers = pd.read_csv(TRAINING_DATA_FILE_PATH, nrows=0).columns.tolist()
# Then read the data, using the header to key the columns
data = pd.read_csv(TRAINING_DATA_FILE_PATH, usecols=headers, sep=",")

# Convert each column of data to compatible tensors
data_x = torch.tensor(data["length"].values.tolist()).reshape(-1, 1)
data_y = torch.tensor(data["weight"].values.tolist()).reshape(-1, 1)

# data_x = torch.tensor([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(-1, 1)
# data_y = torch.tensor([5.0, 3.5, 3.0, 4.0, 3.0, 1.5, 2.0]).reshape(-1, 1)

model = LinearRegressionModel()
optimizer = torch.optim.adam([model.W, model.b], 0.00015)

for epoch in range(1000):
    model.loss(data_x, data_y).backward()
    optimizer.step()
    optimizer.zero_grad()

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
