import pickle

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import ConfusionMatrixDisplay


class ConfusionMatrixTracker:
    MEASURE_TYPE = "confusion-matrix"

    def __init__(self, benchmark):
        self.benchmark = benchmark        

    def track(self, matrix, labels, description):
        serialized = self.serialize(matrix, labels)
        self.benchmark.log(description, self.MEASURE_TYPE, serialized)

    def serialize(self, matrix, labels):
        return pickle.dumps({'matrix': matrix, 'labels': labels})


class ConfusionMatrixVisualizer:
    def __init__(self, serialized_bytes):
        deserialized = pickle.loads(serialized_bytes)
        self.matrix = deserialized['matrix']
        self.labels = deserialized['labels']

    def visualize(self):
        matrix_str = [[str(y) for y in x] for x in self.matrix]
        fig = ff.create_annotated_heatmap(self.matrix, 
                                          x=self.labels,
                                          y=self.labels,
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

    def visualize(self):
        color_scale = px.colors.diverging.Tealrose
        if not self.low_means_good:
            color_scale = list(reversed(color_scale))

        fig = px.parallel_coordinates(self.df, 
                                      color=self.target,
                                      dimensions=self.hyperparameters,
                                      color_continuous_scale=color_scale)
        fig.show()