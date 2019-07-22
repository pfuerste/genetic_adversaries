import utils, fileutils, paths
import numpy as np
import librosa
import geneticsearch, model
import keras
import tensorflow as tf

#fileutils.save_data_to_array(path=paths.get_dummy_path(), input_shape=(40, 98, 1))

tf.set_random_seed(0)

X_train, X_test, y_train, y_test = utils.get_train_test(path=paths.get_data_path(), input_shape=(40, 98, 1))
X_train, X_test, y_train_hot, y_test_hot = utils.reshape_data((40, 98, 1), X_train, X_test, y_train, y_test)
print(np.shape(X_train), np.shape(y_train))
print(np.max(y_train))

#model = model.Model((40, 98, 1), 2)
#model.model.fit(X_train, y_train_hot, batch_size=64, epochs=1, verbose=1, validation_data=(X_test, y_test_hot))
#model.model.save('model2.h5')
model = model.Model((40, 98, 1), 4)
cb = keras.callbacks.ModelCheckpoint('model4.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

#model.model.load_weights('cb_w4_e70_bs64_c30.h5')
model.model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adagrad(),
                      metrics=['accuracy'])
model.model.fit(X_train, y_train_hot, batch_size=64, epochs=30, verbose=1, validation_data=(X_test, y_test_hot), callbacks=[cb])
#model.model.save('model4.h5')
loss, acc = model.model.evaluate(x=X_test, y=y_test_hot)
print(loss, acc)
#print(model.model.metrics_names)
#geneticsearch = geneticsearch.GeneticSearch(model=model, filepath=path,
#                                           epochs=100, nb_parents=8, mutation_rate=0.0015,
#                                           popsize=12)
#geneticsearch.search('.', verbose=0)
