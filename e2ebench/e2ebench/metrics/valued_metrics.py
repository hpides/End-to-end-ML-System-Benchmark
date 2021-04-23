import pickle

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import ConfusionMatrixDisplay

class Visualizer:
    def __init__(self, df):
        assert 'uuid' in df.columns
        assert 'type' in df.columns
        assert 'desc' in df.columns
        assert 'bytes' in df.columns
        self.df = df

class ConfusionMatrixTracker:
    MEASURE_TYPE = "confusion-matrix"

    def __init__(self, benchmark):
        self.benchmark = benchmark        

    def track(self, matrix, labels, description):
        serialized = self.serialize(matrix, labels)
        self.benchmark.log(description, self.MEASURE_TYPE, serialized)

    def serialize(self, matrix, labels):
        return pickle.dumps({'matrix': matrix, 'labels': labels})


class ConfusionMatrixVisualizer(Visualizer):
    def visualize(self):
        for _, row in self.df.iterrows():
            deserialized = pickle.loads(row['bytes'])
            matrix = deserialized['matrix']
            labels = deserialized['labels']
            matrix_str = [[str(y) for y in x] for x in matrix]
            fig = ff.create_annotated_heatmap(matrix, 
                                            x=labels,
                                            y=labels,
                                            annotation_text=matrix_str,
                                            colorscale=px.colors.diverging.Tealrose
                                            )

            layout = {
                "xaxis" : {"title" : "Predicted Value"},
                "yaxis" : {"title" : "Real Value"},
            }

            fig.show()

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

class HyperparameterVisualizer:
    def __init__(self, serialized_bytes):
        deserialized = pickle.loads(serialized_bytes)
        self.hyperparameters = deserialized['hyperparameters']
        self.df = pd.DataFrame(deserialized['df'])
        self.target = deserialized['target']
        self.low_means_good = deserialized['low_means_good']

    def visualize(self, uuid, description, starts):
        color_scale = px.colors.diverging.Tealrose
        if not self.low_means_good:
            color_scale = list(reversed(color_scale))

        fig = px.parallel_coordinates(self.df, 
                                      color=self.target,
                                      dimensions=self.hyperparameters,
                                      color_continuous_scale=color_scale)
        fig.show()


class TTATracker:
    MEASURE_TYPE = "tta"

    def __init__(self, benchmark):
        self.benchmark = benchmark

    def track(self, accuracies, description):
        serialized = self.serialize(accuracies)
        self.benchmark.log(description, self.MEASURE_TYPE, serialized)

    def serialize(self, accuracies):
        return pickle.dumps({'accuracies': accuracies})


class TTAVisualizer:
    def __init__(self, serialized_bytes):
        self.accuracies = []
        for values in serialized_bytes:
            self.accuracies.append(pickle.loads(values)['accuracies'])

    def visualize(self, uuid, description, starts):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        for run in range(len(self.accuracies)):
            x_values = []
            for i in range(len(self.accuracies[run])):
                x_values.append(i + 1)
            plt.xticks(rotation=90)
            ax.plot(['{:.1f}'.format(x) for x in x_values],
                self.accuracies[run],
                label=("Run from " + str(starts[run].isoformat(' ', 'seconds'))))

        plt.legend(loc=2)
        ax.set_ylabel("accuracy")
        ax.set_xlabel("epoch")
        plt.title("Time to accuracy")

        ax.yaxis.set_major_locator(ticker.LinearLocator(12))
        plt.show()


class LossTracker:
    MEASURE_TYPE = "loss"

    def __init__(self, benchmark):
        self.benchmark = benchmark

    def track(self, loss, description):
        serialized = self.serialize(loss)
        self.benchmark.log(description, self.MEASURE_TYPE, serialized)

    def serialize(self, accuracies):
        return pickle.dumps({'loss': accuracies})


class LossVisualizer:
    def __init__(self, serialized_bytes):
        self.loss = []
        for values in serialized_bytes:
            self.loss.append(pickle.loads(values)['loss'])

    def visualize(self, uuid, description, starts):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        for run in range(len(self.loss)):
            x_values = []
            for i in range(len(self.loss[run])):
                x_values.append(i + 1)
            plt.xticks(rotation=90)
            ax.plot(['{:.1f}'.format(x) for x in x_values],
                self.loss[run],
                label=("Run from " + str(starts[run].isoformat(' ', 'seconds'))))

        plt.legend(loc=2)
        ax.set_ylabel("loss")
        ax.set_xlabel("epoch")
        plt.title("Training loss")

        ax.yaxis.set_major_locator(ticker.LinearLocator(12))
        plt.show()
