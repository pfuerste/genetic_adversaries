from model import Model
import itertools
import keras.optimizers
import numpy as np
from utils import get_train_test, reshape_data
from paths import get_dummy_path, get_small_path


class GridSearch:
    def __init__(self, path, input_shape, verbose=1, **kwargs):
        self.input_shape = input_shape
        self.verbose = verbose
        self.hist = dict()
        self.hyperparams = check_params(**kwargs)
        self.permutations = self.get_perm()
        self.path = path
        self.data = reshape_data(input_shape, *get_train_test(path=path, input_shape=input_shape))

    def get_perm(self):
        array = list(self.hyperparams.values())
        permutations = list(itertools.product(*array))
        return permutations

    def search(self):

        x_train, x_test, y_train_hot, y_test_hot = self.data

        print('Performing Hyperparametersearch over {}, this will take {} runs.'.format(
            self.hyperparams, len(self.permutations)))

        for index, instance in enumerate(self.permutations):
            model_version, optimizer, batch_size, epochs = instance

            model = Model(self.input_shape, model_version, self.path)
            model.model.compile(loss=keras.losses.categorical_crossentropy,
                                optimizer=get_optimizer(optimizer),
                                metrics=['accuracy'])
            hist = model.model.fit(x_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=self.verbose,
                                   validation_data=(x_test, y_test_hot))
            # Log instance params and acc & val_acc at epoch with highest val_acc
            self.hist[index] = dict(hyperparams=instance, peak_epoch=np.argmax(hist.history['val_acc']) + 1,
                                    acc=hist.history['acc'][np.argmax(hist.history['val_acc'])],
                                    val_acc=np.max(hist.history['val_acc']))

    def get_log(self):
        if not self.hist:
            print('Perform Search before looking up Results.')
        for run in self.hist.keys():
            print(self.hist.get(run))

    def get_best_run(self):
        if not self.hist:
            print('Perform Search before looking up Results.')
        val_accs = list()
        keys = self.hist.keys()
        for key in keys:
            val_accs.append(self.hist.get(key).get('val_acc'))
        max_index = np.argmax(val_accs)
        print('Best performing run {}: val_acc: {}, acc: {}, instance params: {}'
              .format(max_index, np.max(val_accs), self.hist.get(max_index).get('acc'),
                      self.hist.get(max_index).get('hyperparams')))


def get_optimizer(optimizer):
    optimizers = {'adadelta': keras.optimizers.Adadelta(), 'sgd': keras.optimizers.SGD(),
                  'adam': keras.optimizers.Adam(), 'adagrad': keras.optimizers.Adagrad()}
    return optimizers.get(optimizer)


def check_params(**kwargs):
    hyperparams = dict()
    for k in kwargs.keys():
        if k in ['version', 'optimizer', 'batch_size', 'epochs']:
            hyperparams[k] = kwargs[k]
    if not len(kwargs.keys()) == 4:
        print('Please specify version, optimizer, batch size and epochs.')
    return hyperparams


if __name__ == "__main__":

    search = GridSearch(get_small_path(), (40, 98, 1), version=[3, 4], optimizer=['adadelta'], batch_size=[64], epochs=[50])
    search.search()
    search.get_log()
    search.get_best_run()
