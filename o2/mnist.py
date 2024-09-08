import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

TRAINING_DATA_FILE_PATH = "o2/data/mnist_train.csv"
TEST_DATA_FILE_PATH = "o2/data/mnist_test.csv"

mnist_data_training = pd.read_csv(TRAINING_DATA_FILE_PATH, sep=",")
mnist_data_test = pd.read_csv(TEST_DATA_FILE_PATH, sep=",", header=None)

training_labels = mnist_data_training[0]
training_images = mnist_data_training.loc[:, mnist_data_training.columns != 0]

test_labels = mnist_data_test[0]
test_images = mnist_data_test.loc[:, mnist_data_test.columns != 0]


plt.show()
