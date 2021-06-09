import h5py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from math import floor
from realTimeBench.realTimeBenchmarker import realTimeBenchmarker


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

    for i in range(10):
        input_train = train_f['sen1'][floor(352366*i/10):floor(352366*(i + 1)/10) - 1]
        label_train = train_f['label'][floor(352366*i/10):floor(352366*(i + 1)/10) - 1]
        input_val = val_f['sen1'][floor(24119*i/10):floor(24119*(i + 1)/10) - 1]
        label_val = val_f['label'][floor(24119*i/10):floor(24119*(i + 1)/10) - 1]

        input_train = input_train.reshape((len(input_train), img_width, img_height, img_num_channels))

        history = model.fit(input_train, label_train,
                            batch_size=batch_size,
                            epochs=no_epochs,
                            verbose=0,
                            validation_data=(input_val, label_val), callbacks=[realTimeBenchmarker(
                 input_size= len(input_train), batch_size= batch_size, no_epochs= no_epochs
            )])

        print(i)
        print("LOSS:", history.history['loss'])

    train_f.close()
    val_f.close()

    return model
