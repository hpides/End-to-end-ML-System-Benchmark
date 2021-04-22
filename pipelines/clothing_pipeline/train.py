import tensorflow as tf
from e2ebench import Benchmark, ConfusionMatrixTracker, HyperparameterTracker, BenchmarkSupervisor, TimeMetric, \
    MemoryMetric, PowerMetric, EnergyMetric, LatencyMetric, ThroughputMetric, TTATracker, LossTracker

bm = Benchmark('clothing.db')
lat = LatencyMetric('clothing latency')
thr = ThroughputMetric('clothing throughput')
tta = TTATracker(bm)
loss = LossTracker(bm)


@BenchmarkSupervisor([TimeMetric('clothing time'), MemoryMetric('clothing memory', interval=0.1),
                      PowerMetric('clothing power'), EnergyMetric('clothing energy'),
                      lat, thr], bm)
def train(train_images, train_labels):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer="Adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=100, batch_size=256)

    lat.track(num_entries=len(train_images))
    thr.track(num_entries=len(train_images))
    tta.track(accuracies=history.history["accuracy"],  description="clothing TTA")
    loss.track(loss=history.history['loss'], description="clothing loss")

    return model
