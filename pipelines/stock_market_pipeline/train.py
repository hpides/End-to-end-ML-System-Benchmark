from benchmarking import bm
import e2ebench
from tensorflow.keras.optimizers import Adam


@e2ebench.MeasureLearningRate(bm, description="Learning Rate")
#@e2ebench.MeasureBatchAndEpochInfluence(bm, description="Batch Size and Epoch Influence")
#@e2ebench.MeasureBatchSizeInfluence(bm, description="Batch Size Influence")
#@e2ebench.MeasureThroughput(bm, description="Training throughput")
#@e2ebench.MeasureLatency(bm, description="Training latency")
#@e2ebench.MeasureLoss(bm, description="Training loss")
#@e2ebench.MeasureTimeToAccuracy(bm, description="Time to Accuracy")
#@e2ebench.MeasureTime(bm, description="Training Time")
#@e2ebench.MeasureEnergy(bm, description="Training Energy usage")
def train(model, X_train, y_train, batch_size=32, epochs=10, lr=0.01):

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics=["accuracy", "mse"])

    history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)

    return {"model": model, "num_entries": len(X_train),
            "accuracy": history.history["accuracy"], "loss": history.history["loss"]}
