import pickle

import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

class ConfusionMatrixTracker:
    MEASURE_TYPE = "confusion-matrix"

    def __init__(self, benchmark=None, matrix=None, labels=None, description=None):
        self.benchmark = benchmark
        self.matrix = matrix
        self.labels = labels
        self.description = description

    @classmethod
    def _from_serialized(cls, serialized):
        deserialized = pickle.loads(serialized)
        return cls(matrix=deserialized['matrix'], labels=deserialized['labels'])

    def track(self, matrix, labels, description):
        serialized = self.serialize(matrix, labels)
        self.benchmark.log(description, self.MEASURE_TYPE, serialized)

    def serialize(self, matrix, labels):
        return pickle.dumps({'matrix': matrix, 'labels': labels})

    def visualize(self):
        display = ConfusionMatrixDisplay(confusion_matrix=self.matrix, display_labels=self.labels)
        display.plot()

class HyperparameterTracker:
    MEASURE_TYPE = "hyperparameters"

    MEASUREMENT_COLUMNS = ["training_id", "epoch_id", "loss", "learning_rate", "batch_size"]

    def __init__(self, benchmark=None, measurements=None, description=None):
        self.benchmark = benchmark
        self.measurements = defaultdict(list)
        self.description = description

    @classmethod
    def _from_serialized(cls, serialized):
        deserialized = pickle.loads(serialized)
        return cls(measurements=deserialized)

    def track(self, training_id, epoch_id, loss, learning_rate=None, batch_size=None):
        self.measurements["training_id"].append(training_id)
        self.measurements["epoch_id"].append(epoch_id)
        self.measurements["loss"].append(loss)
        self.measurements["learning_rate"].append(loss)
        self.measurements["batch_size"].append(batch_size)
    
    def close(self):
        serialized = self.serialize(self.measurements)
        self.benchmark.log(serialized)

    def serialize(self):
        return pickle.dumps(self.measurements)

    def visualize(self):
        pass
