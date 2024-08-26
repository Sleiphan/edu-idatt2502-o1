import torch


class LinearRegressionModel:
    def __init__(self, initial_W: float = 0.0, initial_b: float = 0.0):
        self.W = torch.tensor([[initial_W]], requires_grad=True)
        self.b = torch.tensor([[initial_b]], requires_grad=True)

    def predict(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.predict(x) - y))
