import os
import sys

import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import e2ebench

from benchmarking import bm

@e2ebench.BenchmarkSupervisor([e2ebench.TimeMetric('training time')], bm)
def train():
    with h5py.File('data/h5py.h5', 'r') as hdf:
        X_train = hdf['X_train'][:,:]
        y_train = hdf['y_train'][:]
        classifier = RandomForestClassifier(n_estimators=10, max_depth=2, min_samples_split=100, random_state=0)
        classifier.fit(X_train, y_train)
        return {'num_entries': len(X_train), 'classifier': classifier}


def test(training_result):
    with h5py.File('data/h5py.h5', 'r') as hdf:
        classifier = training_result['classifier']
        X_test = hdf['X_test'][:,:]
        y_test = hdf['y_test'][:]
        y_pred = classifier.predict(X_test)

        conf_mat = confusion_matrix(y_test, y_pred)
        labels = classifier.classes_

        e2ebench.ConfusionMatrixTracker(bm).track(conf_mat, labels, "Testing results confusion matrix")


def train_and_test():
    classifier = train()
    test(classifier)


if __name__ == "__main__":
    train_and_test()