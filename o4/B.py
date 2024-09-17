import torch
import torch.nn as nn
import numpy as np


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


emojis = {
    'hat': '\U0001F3A9',
    'rat': '\U0001F400',
    'cat': '\U0001F408',
    'flat': '\U0001F3E2',
    'matt': '\U0001F468',
    'cap': '\U0001F9E2',
    'son': '\U0001F466'
}

index_to_word = ['hat', 'rat', 'cat', 'flat', 'matt', 'cap', 'son']
index_to_char = set(' '.join(index_to_word))

char_encodings = np.eye(len(index_to_char), dtype=float)
emoji_encodings = np.eye(len(index_to_word), dtype=float)

chars = {}
for i, char in enumerate(index_to_char):
    chars[char] = char_encodings[i]

x_train = torch.tensor(np.array([
    [[chars['h']], [chars['a']], [chars['t']], [chars[' ']]],
    [[chars['r']], [chars['a']], [chars['t']], [chars[' ']]],
    [[chars['c']], [chars['a']], [chars['t']], [chars[' ']]],
    [[chars['f']], [chars['l']], [chars['a']], [chars['t']]],
    [[chars['m']], [chars['a']], [chars['t']], [chars['t']]],
    [[chars['c']], [chars['a']], [chars['p']], [chars[' ']]],
    [[chars['s']], [chars['o']], [chars['n']], [chars[' ']]]
]), dtype=torch.float32)

y_train = torch.tensor(
    np.repeat([emoji_encodings], 4, axis=1).reshape(-1, 4, len(emoji_encodings)))


model = LongShortTermMemoryModel(len(char_encodings), len(emoji_encodings))


optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(500):
    for i in range(x_train.size()[0]):
        model.reset()
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()


def to_emoji(text: str):
    model.reset()
    for char in text:
        out = model.f(torch.tensor(
            [[chars[char]]], dtype=torch.float))
    return emojis[index_to_word[out.argmax(1)]]


test_strings = ['rt', 'rats', 'ma', 'c', 'ca']

for s in test_strings:
    print(f'{s} -> {to_emoji(s)}')
print()

for word in index_to_word:
    print(f'{word} -> {to_emoji(word)}')
