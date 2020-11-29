import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

traindata_path = os.getcwd() + "/data/normalized.csv"


def train_and_test():
    df = pd.read_csv(traindata_path)
    dataset = df.values

    # Create the feature dataset X (date / serial number not yet included, needs to be embedded first)
    X = dataset[:, 2:6]
    X = np.asarray(X).astype('float32')

    # Create the label dataset (1-5 days to fail columns)
    Y = dataset[:,6:11]
    Y = np.asarray(Y).astype('float32')

    # Split into training, validation and test dataset
    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

    #  X_train / Y_train (70% of full dataset)     Training Set
    #  X_val / Y_val (15% of full dataset)         Validation Set
    #  X_test / Y_test (15% of full dataset)       Test Set

    print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

    # Create the NN, 3 layers, 1st layer 4 nodes (features), 2nd layer 16 nodes, 3rd layer 5 nodes (predicted labels)
    model = Sequential([Dense(32, activation='relu', input_shape=(4,)), Dense(32, activation='relu'),
                        Dense(5, activation='sigmoid'), ])
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    # training
    hist = model.fit(X_train, Y_train, batch_size=32, epochs=5, validation_data=(X_val, Y_val))

    # testing
    print("Testing loss and accuracy:")
    print(model.evaluate(X_test, Y_test))

    # plot the loss during the training phase
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()

    # plot the accuracy during the training phase
    # plt.plot(hist.history['acc'])
    # plt.plot(hist.history['val_acc'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Val'], loc='lower right')
    # plt.show()


if __name__ == "__main__":
    train_and_test()
