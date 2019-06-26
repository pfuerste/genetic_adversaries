from keras.utils import to_categorical

from model import Model
from paths import get_dummy_path, pick_random_sample
from utils import get_train_test

import numpy as np

# Feature dimension
input_shape = (98, 40, 1)
epochs = 1
batch_size = 1
verbose = 1
num_classes = 30


# Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test(path=get_dummy_path())

# Reshaping to perform 2D convolution
X_train = X_train.reshape(X_train.shape[0], input_shape[0], input_shape[1], input_shape[2])
X_test = X_test.reshape(X_test.shape[0], input_shape[0], input_shape[1], input_shape[2])

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_train_hot))
print(np.shape(y_test_hot))

model = Model(input_shape)
model.model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))

print(model.predict(pick_random_sample()))