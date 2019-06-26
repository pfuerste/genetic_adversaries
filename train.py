from typing import Tuple

from utils import *
from model import get_model, predict
from keras.utils import to_categorical
from fileutils import pick_random_sample

# Feature dimension
input_shape = (98, 40, 1)
epochs = 50
batch_size = 100
verbose = 1
num_classes = 30


# Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test(path=DUMMY_PATH)

# Reshaping to perform 2D convolution
X_train = X_train.reshape(X_train.shape[0], input_shape[0], input_shape[1], input_shape[2])
X_test = X_test.reshape(X_test.shape[0], input_shape[0], input_shape[1], input_shape[2])

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

model = get_model(input_shape=input_shape)
model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))

print(predict(pick_random_sample(), model, input_shape))