import os
import sys
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import package as pkg
from benchmarking import bm

sys.path.insert(0, os.getcwd())


@pkg.MeasureThroughput(bm, description="Training throughput")
@pkg.MeasureTime(bm, description="Training time")
@pkg.MeasureMemoryPsutil(bm, description="Training memory")
def train():
    with h5py.File('data/h5py.h5', 'r') as hdf:
        X_train = hdf['X_train'][:,:]
        y_train = hdf['y_train'][:]
        classifier = RandomForestClassifier(n_estimators=10, max_depth=2, min_samples_split=100, random_state=0)
        classifier.fit(X_train, y_train)
        return {'num_entries': len(X_train), 'classifier': classifier}


@pkg.MeasureMulticlassConfusion(bm, description="Testing/Validation results")
@pkg.MeasureMemoryPsutil(bm, description="Testing/Validation results")
def test(training_result):
    with h5py.File('data/h5py.h5', 'r') as hdf:
        classifier = training_result['classifier']
        X_test = hdf['X_test'][:,:]
        y_test = hdf['y_test'][:]
        y_pred = classifier.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        classes = classifier.classes_
        # ConfusionMatrixDisplay(conf_mat, classes).plot()
        return {"confusion matrix": conf_mat, "classes": classes}


def train_and_test():
    classifier = train()
    test(classifier)


if __name__ == "__main__":
    train_and_test()