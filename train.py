
from model import Model
from paths import get_small_path, get_data_path, pick_random_sample
from utils import get_train_test, reshape_data
import keras
import numpy as np

# Feature dimension
input_shape = (40, 98, 1)
epochs = 5
batch_size = 64
verbose = 1
num_classes = 10


# Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test(path=get_small_path(), input_shape=input_shape)
X_train, X_test, y_train_hot, y_test_hot = reshape_data(input_shape,X_train, X_test, y_train, y_test)


model = Model(input_shape, 3, path=get_small_path())
model.model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
model.model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))
model.model.save('model4_test.h5')
