import pickle

import pandas as pd

class ConfusionMatrixTracker:
    MEASURE_TYPE = "confusion-matrix"

    def __init__(self, benchmark):
        self.benchmark = benchmark        

    def track(self, matrix, labels, description):
        """
        Pass an ndarray where axis 0 is predicted and axis 1 is actual.
        """
        serialized = self.serialize(matrix, labels)
        self.benchmark.log(description, self.MEASURE_TYPE, serialized)

    def serialize(self, matrix, labels):
        return pickle.dumps({'matrix': matrix, 'labels': labels})

class HyperparameterTracker:
    MEASURE_TYPE = "hyperparameters"

    def __init__(self, benchmark, description, hyperparameters, target, low_means_good=True):
        self.benchmark = benchmark
        self.description = description
        self.hyperparameters = hyperparameters
        self.target = target
        self.low_means_good = low_means_good

        self.df = pd.DataFrame(columns=hyperparameters + [target])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def track(self, measurement):
        for param in self.hyperparameters:
            assert param in measurement, f"Hyperparameter {param} not found in given measurement."
        assert self.target in measurement, f"Target variable {self.target} not found in given measurement."

        measurement = {k: v for k, v in measurement.items() if k in self.hyperparameters + [self.target]}
        self.df.loc[len(self.df)] = measurement
    
    def close(self):
        serialized = self.serialize()
        self.benchmark.log(self.description, self.MEASURE_TYPE, serialized)

    def serialize(self):
        return pickle.dumps({
            'hyperparameters' : self.hyperparameters,
            'df' : self.df.to_dict(orient='list'),
            'target' : self.target,
            'low_means_good' : self.low_means_good
        })

class TTATracker:
    MEASURE_TYPE = "tta"

    def __init__(self, benchmark):
        self.benchmark = benchmark

    def track(self, accuracies, description):
        serialized = self.serialize(accuracies)
        self.benchmark.log(description, self.MEASURE_TYPE, serialized, unit='accuracy')

    def serialize(self, accuracies):
        return pickle.dumps(accuracies)

class LossTracker:
    MEASURE_TYPE = "loss"

    def __init__(self, benchmark):
        self.benchmark = benchmark

    def track(self, loss_values, description):
        serialized = self.serialize(loss_values)
        self.benchmark.log(description, self.MEASURE_TYPE, serialized, unit="loss")

    def serialize(self, loss_values):
        return pickle.dumps(loss_values)