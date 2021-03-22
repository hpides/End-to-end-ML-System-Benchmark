from e2ebench.metrics.valued_metrics import ConfusionMatrixTracker, HyperparameterTracker
from e2ebench.metrics.supervisor import BenchmarkSupervisor, TimeMetric, MemoryMetric


measurement_type_mapper = {
    "confusion-matrix" : ConfusionMatrixTracker,
    "hyperparameters" : HyperparameterTracker
}