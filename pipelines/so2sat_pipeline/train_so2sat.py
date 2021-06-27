import h5py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from math import floor, ceil
from realTimeBench.realTimeBenchmarker import RtBenchTensorflow
import numpy as np
from keras.utils import Sequence


class generatorClass(Sequence):

    def __init__(self, x, batch_size):
        self.batch_size = batch_size
        self.file = h5py.File(x, 'r')
        self.max = self.file["sen1"].shape[0]
        self.x = self.file["sen1"]
        #self.x = self.x.reshape((len(self.x), 32, 32, 8))
        self.y = self.file["label"]

    def __len__(self):
        return ceil(2**13 / self.batch_size)
        return ceil(self.file["sen1"].shape[0] / self.batch_size)

    def __getitem__(self, idx):
        end = reshape = self.batch_size
        if (idx+self.batch_size > self.max):
            end = self.max
            reshape = reshape - (self.max - idx+self.batch_size)
        return self.x[idx:idx+end].reshape(reshape, 32, 32, 8), self.y[idx:idx+end]


class generator:
    def __init__(self, file):
        self.file = file

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for image, label in zip(hf["sen1"], hf["label"]) :
                yield image, label


#@eb.BenchmarkSupervisor([eb.MemoryMetric('train memory'), eb.TimeMetric('train time'), eb.PowerMetric('train power')], bm)
def train():

    # Model configuration
    batch_size = 256
    img_width, img_height, img_num_channels = 32, 32, 8
    loss_function = "categorical_crossentropy"
    no_classes = 17
    no_epochs = 10
    optimizer = Adam()
    verbosity = 1

    input_shape = (img_width, img_height, img_num_channels)

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))

    model.summary()

    # Compile the model
    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    train_f = h5py.File('/media/jonas/DATA/So2Sat/m1483140/training.h5', 'r')
    val_f = h5py.File('/media/jonas/DATA/So2Sat/m1483140/validation.h5', 'r')

    #f = open('/media/jonas/DATA/So2Sat/m1483140/training.h5', 'r')
    #f2 = open('/media/jonas/DATA/So2Sat/m1483140/validation.h5', 'r')

    #g = generator(f)
    #print(g.next())

    #train_x = np.asarray(train_f["sen1"])
    #train_y = np.asarray(train_f["label"])
    #validation_x = np.asarray(val_f["sen1"])
    #validation_y = np.asarray(val_f["label"])


    #g = generatorClass('/media/jonas/DATA/So2Sat/m1483140/training.h5')
    #print(g)
    #print(g.__getitem__(0))

    train_gen = generatorClass('/media/jonas/DATA/So2Sat/m1483140/training.h5', batch_size=2048)
    val_gen = generatorClass('/media/jonas/DATA/So2Sat/m1483140/validation.h5', batch_size=2048)



    model.fit_generator(generator=train_gen,
                        validation_data=val_gen,
                        epochs=10, callbacks=[RtBenchTensorflow(
                        input_size= len(train_gen) + len(val_gen), batch_size=1, no_epochs=no_epochs
                        )])

    #for i in range(10):
    #    input_train = train_f['sen1'][floor(352366*i/10):floor(352366*(i + 1)/10) - 1]
    #    label_train = train_f['label'][floor(352366*i/10):floor(352366*(i + 1)/10) - 1]
    #    input_val = val_f['sen1'][floor(24119*i/10):floor(24119*(i + 1)/10) - 1]
    #    label_val = val_f['label'][floor(24119*i/10):floor(24119*(i + 1)/10) - 1]

    #    input_train = input_train.reshape((len(input_train), img_width, img_height, img_num_channels))

    #    history = model.fit(input_train, label_train,
    #                        batch_size=batch_size,
    #                        epochs=no_epochs,
    #                        verbose=0,
    #                        validation_data=(input_val, label_val), callbacks=[RtBenchTensorflow(
    #             input_size= len(input_train), batch_size= batch_size, no_epochs= no_epochs
    #        )])

    #    print(i)
    #    print("LOSS:", history.history['loss'])

    train_f.close()
    val_f.close()

    return model
