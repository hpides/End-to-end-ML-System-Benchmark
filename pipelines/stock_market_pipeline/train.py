from benchmarking import bm
from tensorflow.keras.optimizers import Adam
import os
import sys
import e2ebench as eb
from benchmarking import bm


@eb.BenchmarkSupervisor([eb.MemoryMetric('train memory'), eb.TimeMetric('train time'), eb.PowerMetric('train power')], bm)
def train(model, X_train, y_train, batch_size=32, epochs=10, lr=0.01):

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics=["accuracy", "mse"])

    history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)

    return {"model": model, "num_entries": len(X_train),
            "accuracy": history.history["accuracy"], "loss": history.history["loss"]}
