import torch
import torch.nn as nn
import torch.optim as optim


class SigmoidModel:
    def __init__(self, W=torch.tensor([[10.]], requires_grad=True), b=torch.tensor([[-5.]], requires_grad=True)):
        self.W = W
        self.b = b
        self.sigmoid = torch.nn.Sigmoid()

    # Predictor
    def forward(self, x):
        return self.sigmoid(x @ self.W + self.b)

    # Uses Cross Entropy
    def loss_CE(self, x, y):
        return -torch.mean(torch.multiply(y, torch.log(self.forward(x))) + torch.multiply((1 - y), torch.log(1 - self.forward(x))))

    # Uses Mean Squared Error
    def loss_MSE(self, x, y):
        return torch.mean(torch.square(y - self.forward(x)))


class SigmoidNandModel(SigmoidModel):
    def __init__(self, W=torch.tensor([[10.], [10.]], requires_grad=True), b=torch.tensor([[-5.]], requires_grad=True)):
        super(SigmoidNandModel, self).__init__(W=W, b=b)

    def forward(self, coords):
        x = coords[:, 0].reshape(-1, 1)
        y = coords[:, 1].reshape(-1, 1)
        W_x = self.W[0, :].reshape(-1, 1)
        W_y = self.W[1, :].reshape(-1, 1)
        return -self.sigmoid(x @ W_x + self.b) * self.sigmoid(y @ W_y + self.b)


class SigmoidXorModel(SigmoidModel):
    def __init__(self, W=torch.tensor([[10.], [10.]], requires_grad=True), b=torch.tensor([[-5.]], requires_grad=True)):
        super(SigmoidXorModel, self).__init__(W=W, b=b)
        self.layer_1 = SigmoidModel(W, b)
        self.layer_2 = SigmoidModel(W, b)

    def forward(self, coords):
        return self.layer_1.forward(self.layer_2.forward(coords))


class SigmoidXorModel:
    def __init__(self,
                 W1=torch.tensor([[10.0, -10.0], [10.0, -10.0]],
                                 requires_grad=True),
                 W2=torch.tensor([[10.0], [10.0]], requires_grad=True),
                 b1=torch.tensor([[-5.0, 15.0]], requires_grad=True),
                 b2=torch.tensor([[-15.0]], requires_grad=True)):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2
        self.sigmoid = torch.nn.Sigmoid()

    # First layer function
    def f1(self, x):
        return self.sigmoid(x @ self.W1 + self.b1)

    # Second layer function
    def f2(self, h):
        return self.sigmoid(h @ self.W2 + self.b2)

    # Predictor
    def forward(self, x):
        return self.f2(self.f1(x))

    # Uses Cross Entropy
    def loss(self, x, y):
        return -torch.mean(torch.multiply(y, torch.log(self.f(x))) + torch.multiply((1 - y), torch.log(1 - self.f(x))))
