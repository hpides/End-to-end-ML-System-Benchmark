from e2ebench.metrics.valued_metrics import ConfusionMatrixTracker, HyperparameterTracker, ConfusionMatrixVisualizer, HyperparameterVisualizer
from e2ebench.metrics.supervisor import BenchmarkSupervisor, TimeMetric, MemoryMetric


measurement_type_mapper = {
    "confusion-matrix" : ConfusionMatrixVisualizer,
    "hyperparameters" : HyperparameterVisualizer
}