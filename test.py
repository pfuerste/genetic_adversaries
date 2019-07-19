import utils, fileutils, paths
import numpy as np
import librosa
import geneticsearch, model

#fileutils.save_data_to_array(path=paths.get_dummy_path(), input_shape=(40, 98, 1))

#np.random.seed(0)

#X_train, X_test, y_train, y_test = utils.get_train_test(path=paths.get_data_path(), input_shape=(40, 98, 1))
#X_train, X_test, y_train_hot, y_test_hot = utils.reshape_data((40, 98, 1), X_train, X_test, y_train, y_test)
path = paths.pick_random_sample()
model = model.Model((40, 98, 1), 'testm.h5')
#model.model.load_weights('cb_w4_e70_bs64_c30.h5')
#model.model.fit(X_train, y_train_hot, batch_size=64, epochs=60, verbose=1, validation_data=(X_test, y_test_hot))
#model.model.save('model1_e60_bs64_c30.h5')
#print(model.model.evaluate(x=X_test, y=y_test_hot))
#print(model.model.metrics_names)
geneticsearch = geneticsearch.GeneticSearch(model=model, filepath=path,
                                           epochs=100, nb_parents=8, mutation_rate=0.0015,
                                           popsize=12)
geneticsearch.search('.', verbose=0)
