import utils, fileutils, paths
import numpy as np
import librosa
import geneticsearch, model


#fileutils.save_data_to_array(path=paths.get_dummy_path(), input_shape=(40, 98, 1))
X_train, X_test, y_train, y_test = utils.get_train_test(path=paths.get_dummy_path(), input_shape=(40, 98, 1))



path = paths.pick_random_sample(paths.get_data_path())
model = model.Model((40, 98, 1), 'model4_e80_bs64_c30.h5')
model.model.evaluate(x=X_test[1], y=y_test[1], batch_size=1)
geneticsearch = geneticsearch.GeneticSearch(model=model, filepath=path,
                                            epochs=10, nb_parents=4, mutation_rate=0.0015,
                                            popsize=8)
geneticsearch.search()
