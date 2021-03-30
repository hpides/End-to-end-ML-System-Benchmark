from e2ebench.metrics.valued_metrics import ConfusionMatrixTracker, HyperparameterTracker, ConfusionMatrixVisualizer, HyperparameterVisualizer
from e2ebench.metrics.supervised_metrics import BenchmarkSupervisor, TimeMetric, MemoryMetric, PowerMetric


measurement_type_mapper = {
    "confusion-matrix" : ConfusionMatrixVisualizer,
    "hyperparameters" : HyperparameterVisualizer
}