import torch
import matplotlib.pyplot as plt
from components.OperatorModels import SigmoidModel

# model = NotModel()
model = SigmoidModel(
    W=torch.tensor([[-10.]], requires_grad=True),
    b=torch.tensor([[5.]], requires_grad=True)
)

plt.xlabel('input')
plt.ylabel('output')
x = torch.arange(start=-0.1, end=1.1, step=0.025).reshape(-1, 1)
plt.plot(x, model.forward(x).detach(), label='$f(x) = sigmoid(xW+b)$')
plt.grid(visible=True)
plt.legend()
plt.show()
