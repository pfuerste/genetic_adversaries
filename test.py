import utils, fileutils, paths
import numpy as np

fileutils.save_data_to_array('mfcc', path=paths.get_dummy_path(), max_len=11, n_mfcc=20, Transpose=False)
X_train, X_test, y_train, y_test = utils.get_train_test(path=paths.get_dummy_path(), input_shape=(20,11,1))
sample = X_train[0]
print(np.shape(sample))
utils.visualize_mfcc(sample)

