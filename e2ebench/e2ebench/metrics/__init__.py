from e2ebench.metrics.valued_metrics import ConfusionMatrixTracker, HyperparameterTracker, ConfusionMatrixVisualizer, \
    HyperparameterVisualizer
from e2ebench.metrics.supervised_metrics import BenchmarkSupervisor, TimeMetric, MemoryMetric, PowerMetric, \
    EnergyMetric, TimeVisualizer, MemoryVisualizer, PowerVisualizer, EnergyVisualizer, LatencyMetric, \
    LatencyVisualizer, ThroughputMetric, ThroughputVisualizer



measurement_type_mapper = {
    "confusion-matrix" : ConfusionMatrixVisualizer,
    "hyperparameters" : HyperparameterVisualizer,
    "time": TimeVisualizer,
    "memory": MemoryVisualizer,
    "power": PowerVisualizer,
    "energy": EnergyVisualizer,
    "latency": LatencyVisualizer,
    "throughput": ThroughputVisualizer
}
