import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):

    def __init__(self, in_size, out_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(in_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, out_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        # Shape: (number of layers, batch size, state size)
        zero_state = torch.zeros(1, 1, 128)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(
            x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


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
    # h   a   t   r   c   f   l   m   p   s   o   n
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

x_train = torch.tensor([[char_encodings[0], char_encodings[1], char_encodings[2], char_encodings[12]],
                        [char_encodings[3], char_encodings[1],
                            char_encodings[2], char_encodings[12]],
                        [char_encodings[4], char_encodings[1],
                            char_encodings[2], char_encodings[12]],
                        [char_encodings[5], char_encodings[6],
                            char_encodings[1], char_encodings[2]],
                        [char_encodings[7], char_encodings[1],
                            char_encodings[2], char_encodings[2]],
                        [char_encodings[4], char_encodings[1],
                            char_encodings[8], char_encodings[12]],
                        [char_encodings[9], char_encodings[10], char_encodings[11], char_encodings[12]]])

y_train = torch.tensor([emoji_encodings[0],
                        emoji_encodings[1],
                        emoji_encodings[2],
                        emoji_encodings[3],
                        emoji_encodings[4],
                        emoji_encodings[5],
                        emoji_encodings[6]])

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
