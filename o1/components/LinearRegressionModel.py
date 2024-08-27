import torch


class LinearRegressionModel:
    def __init__(self, initial_W: float = 0.0, initial_b: float = 0.0):
        self.W = torch.tensor([[initial_W]], requires_grad=True)
        self.b = torch.tensor([[initial_b]], requires_grad=True)

    def predict(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        prediction = self.predict(x)
        target = y
        return torch.mean(torch.square(target - prediction))


class MultipleLinearRegression(torch.nn.Module):
    # Constructor
    def __init__(self, input_dim, output_dim):
        super(MultipleLinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    # Prediction
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    @ property
    def intersect(self):
        return self.get_parameter("linear.bias").item()

    @ property
    def coefficients(self):
        return self.get_parameter("linear.weight").tolist()

    def __str__(self):
        result = "B0: " + str(self.intersect)

        coeff_idx = 1
        for c in self.coefficients:
            for c2 in c:
                result += "\nB" + str(coeff_idx) + ": " + str(c2)
                coeff_idx += 1

        return result
