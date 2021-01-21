import h5py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam

def train():

    # Model configuration
    batch_size = 256
    img_width, img_height, img_num_channels = 32, 32, 8
    loss_function = "categorical_crossentropy"
    no_classes = 17
    no_epochs = 10
    optimizer = Adam()
    verbosity = 1

    n = 32768           ## 2**15

    # Load data
    f = h5py.File('/data/training.h5', 'r')
    input_train = f['sen1'][0:n]
    label_train = f['label'][0:n]
    f.close()
    f = h5py.File('/data/validation.h5', 'r')
    input_val = f['sen1'][0:n]
    label_val = f['label'][0:n]
    f.close()

    # Reshape data
    input_train = input_train.reshape((len(input_train), img_width, img_height, img_num_channels))

    # Determine shape of the data
    input_shape = (img_width, img_height, img_num_channels)

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))

    # Display a model summary
    model.summary()

    # Compile the model
    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Fit data to model
    history = model.fit(input_train, label_train,
                batch_size=batch_size,
                epochs=no_epochs,
                verbose=verbosity,
                validation_data=(input_val, label_val))

    return model
