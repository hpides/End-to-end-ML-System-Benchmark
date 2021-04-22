from data_preparation import prep_data
from train import train
from test import test


def main():
    train_images, test_images, train_labels, test_labels = prep_data()
    model = train(train_images, train_labels)
    test(model, test_images, test_labels)


if __name__ == "__main__":
    main()
