import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter

def train_and_test():
    with h5py.File('data/h5py.h5', 'a') as hdf:
        X_train = hdf['X_train'][:,:]
        y_train = hdf['y_train'][:]
        X_test = hdf['X_test'][:,:]
        y_test = hdf['y_test'][:]

        print(X_train.shape)
        print(y_train.shape)
        print(Counter(y_train))
        print(Counter(y_test))

        classifier = RandomForestClassifier(random_state=0)
        classifier.fit(X_train, y_train)
        plot_confusion_matrix(classifier, X_test, y_test)
        plt.show()

if __name__ == "__main__":
    train_and_test()