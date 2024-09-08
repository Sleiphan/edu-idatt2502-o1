import torch
import matplotlib.pyplot as plt
from components.OperatorModels import SigmoidNandModel


model = SigmoidNandModel()

fig = plt.figure()
# 3D plots require a subplot with 3D projection enabled
ax = fig.add_subplot(projection='3d')

# Create the vector [0., 0.01, ..., 0.99]
x = torch.arange(start=0., end=1., step=0.1)
# Create the square matrix [[0., 0.01, ..., 0.99], ..., [0., 0.01, ..., 0.99]]
x = x.expand(x.shape[0], -1)
# Transpose of x: [[0., ..., 0.], [0.01, ..., 0.01], ... [0.99, ..., 0.99]]
y = x.T
# * and + are element-wise operators resulting in a matrix with the same shape as x and y

# Create an tensor containing all unique coordinates we want to cover
coords = torch.stack((x.flatten(), y.flatten()), dim=1)

z = model.forward(coords).reshape(x.shape[0], x.shape[1])

# Convert the tensors to numpy arrays for plotting
x_np = x.detach().numpy()
y_np = y.detach().numpy()
z_np = z.detach().numpy()  # Use detach to remove gradients before converting to numpy

ax.plot_wireframe(
    x_np, y_np, z_np,
    label="NAND")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Output')
plt.legend()
plt.show()
