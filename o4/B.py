import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):

    def __init__(self, in_size, out_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.state_size = 128
        self.lstm = nn.LSTM(in_size, self.state_size)  # 128 is the state size
        # 128 is the state size
        self.dense = nn.Linear(self.state_size, out_size)

    def reset(self):  # Reset states prior to new input sequence
        # Shape: (number of layers, batch size, state size)
        zero_state = torch.zeros(1, 7, self.state_size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(
            x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, self.state_size))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        calculated_logits = self.logits(x)
        calced_argmax = y.argmax(1)
        return nn.functional.cross_entropy(calculated_logits, calced_argmax)


# char_encodings = [
#     # h   a   t   r   f   l   m   c   p   s   o   n
#     [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'hat'  at 0
#     [0., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0.],  # 'rat'  at 1
#     [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'cat'  at 2
#     [0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0.],  # 'flat' at 3
#     [0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'matt' at 4
#     [0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.],  # 'cap'  at 5
#     [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.],  # 'son'  at 6
# ]
char_encodings = [
    # h   a   t   r   c   f   l   m   p   s   o   n  \0
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h' at  0
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a' at  1
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 't' at  2
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'r' at  3
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'c' at  4
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'f' at  5
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 'l' at  6
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'm' at  7
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 'p' at  8
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 's' at  9
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # 'o' at 10
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # 'n' at 11
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],  # '\0' at 12
]
char_encoding_size = len(char_encodings)

emoji_encodings = [
    [1., 0., 0., 0., 0., 0., 0.],  # 'hat'  at 0
    [0., 1., 0., 0., 0., 0., 0.],  # 'rat'  at 1
    [0., 0., 1., 0., 0., 0., 0.],  # 'cat'  at 2
    [0., 0., 0., 1., 0., 0., 0.],  # 'flat' at 3
    [0., 0., 0., 0., 1., 0., 0.],  # 'matt' at 4
    [0., 0., 0., 0., 0., 1., 0.],  # 'cap'  at 5
    [0., 0., 0., 0., 0., 0., 1.],  # 'son'  at 6
]
emoji_encoding_size = len(char_encodings)

index_to_emoji = [
    'hat',  # at 0
    'rat',  # at 1
    'cat',  # at 2
    'flat',  # at 3
    'matt',  # at 4
    'cap',  # at 5
    'son',  # at 6
]

x_train_hat = torch.tensor([[char_encodings[0]], [char_encodings[1]],  [
                           char_encodings[2]],  [char_encodings[12]]])
x_train_rat = torch.tensor([[char_encodings[3]], [char_encodings[1]],  [
                           char_encodings[2]],  [char_encodings[12]]])
x_train_cat = torch.tensor([[char_encodings[4]], [char_encodings[1]],  [
                           char_encodings[2]],  [char_encodings[12]]])
x_train_flat = torch.tensor([[char_encodings[5]], [char_encodings[6]],  [
                            char_encodings[1]],  [char_encodings[2]]])
x_train_matt = torch.tensor([[char_encodings[7]], [char_encodings[1]],  [
                            char_encodings[2]],  [char_encodings[2]]])
x_train_cap = torch.tensor([[char_encodings[4]], [char_encodings[1]],  [
                           char_encodings[8]],  [char_encodings[12]]])
x_train_son = torch.tensor([[char_encodings[9]], [char_encodings[10]], [
                           char_encodings[11]], [char_encodings[12]]])

x_train = torch.cat((x_train_hat,
                     x_train_rat,
                     x_train_cat,
                     x_train_flat,
                     x_train_matt,
                     x_train_cap,
                     x_train_son
                     ),
                    dim=1)

y_train = torch.tensor(emoji_encodings)

model = LongShortTermMemoryModel(char_encoding_size, emoji_encoding_size)


def test_emoji_model(e_model: LongShortTermMemoryModel):
    test_data = ['rt', 'rats']
    for test_text in test_data:
        model.reset()
        for char in test_text:
            model_output = model.f(torch.tensor([[char]]))
        print(index_to_emoji[model_output.argmax(1)])


optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(500):
    model.reset()
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 9:
        test_emoji_model(model)
