from benchmarking import bm
from tensorflow.keras.optimizers import Adam
import os
import sys
import umlaut as eb
from benchmarking import bm

from umlaut import Benchmark, ConfusionMatrixTracker, HyperparameterTracker, BenchmarkSupervisor, TimeMetric, \
    MemoryMetric, PowerMetric, EnergyMetric, LatencyMetric, ThroughputMetric, TTATracker


bm = Benchmark('stock_market.db')

lat = LatencyMetric('stock market latency')
thr = ThroughputMetric('stock market throughput')
tta = TTATracker(bm)

@BenchmarkSupervisor([TimeMetric('stock market time'), MemoryMetric('stock market memory', interval=0.1),
                      PowerMetric('stock market power'), EnergyMetric('stock market energy'),
                      lat, thr], bm)
def train(model, X_train, y_train, batch_size=32, epochs=10, lr=0.01):

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=["accuracy", "mse"])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    lat.track(num_entries=len(X_train))
    thr.track(num_entries=len(X_train))
    tta.track(accuracies=history.history["accuracy"],  description="stock market TTA")

    return model
