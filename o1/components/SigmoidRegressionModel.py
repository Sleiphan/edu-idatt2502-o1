import torch


class SigmoidRegressionModel:
    def __init__(self, initial_W: float = 0.0, initial_b: float = 0.0):
        self.W = torch.tensor([[initial_W]], requires_grad=True)
        self.b = torch.tensor([[initial_b]], requires_grad=True)
        self.sigmoid = torch.nn.Sigmoid()

    def predict(self, x):
        return 20 * self.sigmoid(x @ self.W + self.b) + 31

    def loss(self, x, y):
        return torch.mean(torch.square(y - self.predict(x)))
