import model
import keras
import tensorflow as tf

from paths import get_dummy_path, get_data_path, pick_random_sample, get_small_path
from utils import get_train_test, reshape_data

import numpy as np


# Feature dimension
input_shape = (13, 100, 1)
epochs = 100
batch_size = 64

# Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test(path=get_small_path(), input_shape=input_shape)
X_train, X_test, y_train_hot, y_test_hot = reshape_data(input_shape,X_train, X_test, y_train, y_test)
print(np.shape(X_train), np.shape(y_train))
print(np.max(y_train))

#model_file = 'models/model3_13x100.h5'
model = model.Model(input_shape, 3)
#model.model.compile(loss=keras.losses.categorical_crossentropy,
#                      optimizer=keras.optimizers.Adadelta(),
#                      metrics=['accuracy'])

cb_m = keras.callbacks.ModelCheckpoint('/net/projects/scratch/winter/valid_until_31_July_2020/pfuerste/models/model3_13x100.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

model.model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test_hot), callbacks=[cb_m])
model.model.evaluate(X_test, y_test_hot)
