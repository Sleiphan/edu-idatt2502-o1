import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd


TRAINING_DATA_FILE_PATH = "data/mnist_train.csv"
TEST_DATA_FILE_PATH = "data/mnist_test.csv"


def load_test_data(square_images: bool = False):
    print("Loading test data... ", end="", flush=True)
    mnist_test_data = pd.read_csv(TEST_DATA_FILE_PATH, sep=",", header=None)
    print("DONE")

    return load_data_common(mnist_csv_dataframe=mnist_test_data, square_images=square_images)


def load_training_data(square_images: bool = False):
    print("Loading training data... ", end="", flush=True)
    mnist_training_data = pd.read_csv(
        TRAINING_DATA_FILE_PATH, sep=",", header=None)
    print("DONE")

    return load_data_common(mnist_csv_dataframe=mnist_training_data, square_images=square_images)


def load_data_common(mnist_csv_dataframe, square_images: bool = False):
    print("Preprocessing data... ", end="", flush=True)

    # Extract labels and images
    labels = torch.tensor(
        mnist_csv_dataframe[0].to_numpy())
    images = torch.tensor(
        mnist_csv_dataframe.loc[:, mnist_csv_dataframe.columns != 0].to_numpy(), dtype=torch.float32)

    # Perform softmax pre-processing on the images
    # images = torch.nn.functional.softmax(input=images, dim=1)

    # Reshape the images to be 28x28
    if square_images:
        images = images.reshape(-1, 1, 28, 28)

    # Gather images and labels into a dataset
    train_dataset = TensorDataset(images, labels)

    print("DONE")
    return train_dataset
