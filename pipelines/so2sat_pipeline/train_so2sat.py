import h5py
import e2ebench as eb
from benchmarking import bm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def load_data(num_samples=0):
    f = h5py.File('data/training.h5', 'r')
    n = num_samples or len(f['label'])
    input_train = f['sen1'][0:n]
    label_train = f['label'][0:n]
    f.close()
    f = h5py.File('data/validation.h5', 'r')
    input_val = f['sen1'][0:len(f['label'])]
    label_val = f['label'][0:len(f['label'])]
    f.close()
    return input_train, label_train, input_val, label_val


def compile_model(input_shape, num_classes, loss_function, optimizer):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])

    model.summary()

    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


@eb.BenchmarkSupervisor([eb.MemoryMetric('train memory'), eb.TimeMetric('train time')], bm)
def train():
    batch_size = 256
    img_width, img_height, img_num_channels = 32, 32, 8
    input_shape = (img_width, img_height, img_num_channels)
    loss_function = "categorical_crossentropy"
    num_classes = 17
    num_epochs = 10
    optimizer = Adam()
    verbosity = 1

    n = 2048
    # n = 0
    input_train, label_train, input_val, label_val = load_data(n)

    input_train = input_train.reshape((len(input_train), img_width, img_height, img_num_channels))
    input_val = input_val.reshape((len(input_val), img_width, img_height, img_num_channels))

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = compile_model(input_shape, num_classes, loss_function, optimizer)

    history = model.fit(input_train, label_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        verbose=verbosity,
                        validation_data=(input_val, label_val))

    # multiple run version
    # return {"model": model, "num_entries": len(input_train), "classifier": optimizer,
    #         "accuracy": history.history["accuracy"][-1]}

    # single run version
    return {"model": model, "num_entries": len(input_train), "classifier": optimizer,
            "accuracy": history.history["accuracy"]}
