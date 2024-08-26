import torch
import matplotlib.pyplot as plt

# Plot 3 random points
plt.plot(torch.rand(3), torch.rand(3), 'o')

# Line plot of x * x
# Create the vector [0., 0.01, ..., 0.99]
x = torch.arange(start=0., end=1., step=0.01)
# x * x is element-wise multiplication resulting in a vector with the same shape as x
plt.plot(x, x * x, label="$x^2$")

plt.xlabel('$x$')  # $$ activates LaTeX math notation
plt.ylabel('$y$')
plt.legend()
plt.show()
