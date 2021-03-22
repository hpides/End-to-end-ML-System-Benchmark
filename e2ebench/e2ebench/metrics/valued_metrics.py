import pickle
from sklearn.metrics import ConfusionMatrixDisplay

class ConfusionMatrixTracker:
    MEASURE_TYPE = "confusion matrix"

    def __init__(self, benchmark=None, matrix=None, labels=None, description=None):
        self.benchmark = benchmark
        self.measure_type = measure_type
        self.matrix = matrix
        self.labels = labels
        self.description = description

    @classmethod
    def _from_serialized(cls, serialized):
        matrix, labels = pickle.loads(serialized)
        return cls(matrix=matrix, labels=labels)

    def track(self, matrix, labels, description):
        serialized = self.serialize(matrix, labels)
        self.benchmark.log(description, self.MEASURE_TYPE, serialized)

    def serialize(self, matrix, labels):
        return pickle.dumps({'matrix': matrix, 'labels': labels})

    def visualize(self):
        display = ConfusionMatrixDisplay(confusion_matrix=self.matrix, display_labels=self.labels)
    