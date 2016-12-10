import numpy as np
import _pickle as pickle
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

image_size = 28
num_labels = 10

def reformat(dataset, labels):
    # dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    dataset = dataset.reshape((-1, image_size, image_size, 1)).astype(np.float32)
    # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

def read_data_sets():
    pickle_file = 'data/notMNIST.pickle'

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)



    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    train = DataSet(train_dataset, train_labels, dtype=dtypes.uint8, reshape=False)
    validation = DataSet(valid_dataset, valid_labels, dtype=dtypes.uint8, reshape=False)
    test = DataSet(test_dataset, test_labels, dtype=dtypes.uint8, reshape=False)

    return Datasets(train=train, validation=validation, test=test)
