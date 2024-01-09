import pickle
import json
import time

import pandas as pd

class ConfusionMatrixTracker:
    MEASURE_TYPE = "confusion-matrix"

    def __init__(self, benchmark):
        """
        Parameters
        ----------
        benchmark : Benchmark
            Benchmark object to store the data in / retrieve it from.
        """
        self.benchmark = benchmark        

    def track(self, matrix, labels, description):
        """
        Pass an ndarray where axis 0 is predicted and axis 1 is actual.

        Parameters
        ----------
        matrix : list of list of ints
            Tracker matrix values.
        labels : list of str
            Class labels of confusion matrix.
        description:
            Description of tracked confusion matrix.
        """
        serialized = self.serialize(matrix, labels)
        self.benchmark.log(description, self.MEASURE_TYPE, serialized)

    def serialize(self, matrix, labels):
        """
        Parameters
        ----------
        matrix : list of list of ints
            Tracker matrix values.
        labels : list of str
            Class labels of confustion matrix.

        Returns
        -------
        pickle object
            Serialized data.
        """
        #return pickle.dumps({'matrix': matrix, 'labels': labels})
        return json.dumps({'matrix': matrix, 'labels': labels}, indent=4, default=str)


class HyperparameterTracker:
    MEASURE_TYPE = "hyperparameters"

    def __init__(self, benchmark, description, hyperparameters, target, low_means_good=True):
        """
        Parameters
        ----------
        benchmark : Benchmark
           Benchmark object to store the data in / retrieve it from.
        description : str
           Description of the data to be tracked and stored.
        hyperparameters : list of str
           Hyperparameters to be tracked.
        target : str
           Target variable of tracked hyperparameters.
        low_means_good : bool
           True if the a low target value is better than a high target value.
        """
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
        """
        Parameters
        ----------
        measurement : list of str
            List of hyperparameters
        """
        for param in self.hyperparameters:
            assert param in measurement, f"Hyperparameter {param} not found in given measurement."
        assert self.target in measurement, f"Target variable {self.target} not found in given measurement."

        measurement = {k: v for k, v in measurement.items() if k in self.hyperparameters + [self.target]}
        self.df.loc[len(self.df)] = measurement
    
    def close(self):
        serialized = self.serialize()
        self.benchmark.log(self.description, self.MEASURE_TYPE, serialized)

    def serialize(self):
        """
        Returns
        -------
        pickle object
            Serialized data.
        """
        #return pickle.dumps({
        #    'hyperparameters' : self.hyperparameters,
        #    'df' : self.df.to_dict(orient='list'),
        #    'target' : self.target,
        #    'low_means_good' : self.low_means_good
        #})
        return {
            'hyperparameters': self.hyperparameters,
            'df': self.df.to_dict(orient='list'),
            'target': self.target,
            'low_means_good': self.low_means_good
        }


class TTATracker:
    MEASURE_TYPE = "tta"

    def __init__(self, benchmark):
        """
        Parameters
        ----------
        benchmark : Benchmark
            Benchmark object to store the data in / retrieve it from.
        """
        self.benchmark = benchmark

    def track(self, accuracies, description):
        """
        Parameters
        ----------
        accuracies : list of ints
            List of tracked accuracies of the run.
        description : str
            Description of tracked TTA.
        """
        serialized = self.serialize(accuracies)
        for i in range(len(serialized) - 1):
            if serialized[i] > serialized[i+1]:
                serialized[i+1] = serialized[i]
        self.benchmark.log(description, self.MEASURE_TYPE, serialized, unit='accuracy')

    def serialize(self, accuracies):
        """
        Parameters
        ----------
        accuracies : list of ints
            List of tracked accuracies of the run.

        Returns
        -------
        pickle object
            Serialized data.
        """
        #return pickle.dumps(accuracies)
        accuracies = accuracies.tolist()
        return accuracies


class LossTracker:
    MEASURE_TYPE = "loss"

    def __init__(self, benchmark):
        """
        Parameters
        ----------
        benchmark : Benchmark
            Benchmark object to store the data in / retrieve it from.
        """
        self.benchmark = benchmark

    def track(self, loss_values, description):
        """
        Parameters
        ----------
        loss_values : list of ints
            List of tracked loss values of the run.
        description : str
            Description of tracked loss.
        """
        serialized = self.serialize(loss_values)
        self.benchmark.log(description, self.MEASURE_TYPE, serialized, unit="loss")

    def serialize(self, loss_values):
        """
        Parameters
        ----------
        loss_values : list of ints
            List of tracked loss values of the run.

        Returns
        -------
        pickle object
            Serialized data.
        """
        #return pickle.dumps(loss_values)
        loss_values = loss_values.tolist()
        return loss_values


class TimedTTATracker:
    MEASURE_TYPE = "timed tta"

    def __init__(self, benchmark, target_acc):
        """
        Parameters
        ----------
        benchmark : Benchmark
            Benchmark object to store the data in / retrieve it from.
        """
        self.benchmark = benchmark
        self.target_acc = target_acc
        self.time = time.perf_counter()
        self.logged = False

    def track(self, accuracy, description):
        """
        Parameters
        ----------
        accuracy : int
            Current accuracy of the run.
        description : str
            Description of tracked timed TTA.
        """
        if accuracy >= self.target_acc and not self.logged:
            self.benchmark.log(description + " (Target: " + str(self.target_acc) + ")", self.MEASURE_TYPE, time.perf_counter() - self.time, unit='time for target accuracy')
            self.logged = True

    def serialize(self, accuracies):
        """
        Parameters
        ----------
        accuracies : list of ints
            List of tracked accuracies of the run.

        Returns
        -------
        pickle object
            Serialized data.
        """
        #return pickle.dumps(accuracies)
        accuracies = accuracies.tolist()
        return accuracies
