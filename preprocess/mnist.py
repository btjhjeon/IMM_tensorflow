import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import utils
from model import imm


def XycPackage():
    """
    Load Dataset and set package.
    """
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    noOfTask = 3
    x = []
    x_ = []
    y = []
    y_ = []
    xyc_info = []

    x.append(np.concatenate((mnist.train.images,mnist.validation.images)))
    y.append(np.concatenate((mnist.train.labels,mnist.validation.labels)))
    x_.append(mnist.test.images)
    y_.append(mnist.test.labels)
    xyc_info.append([x[0], y[0], 'train-idx1'])

    for i in range(1, noOfTask):
        idx = np.arange(784)                 # indices of shuffling image
        np.random.shuffle(idx)
        
        x.append(x[0].copy())
        x_.append(x_[0].copy())
        y.append(y[0].copy())
        y_.append(y_[0].copy())

        x[i] = x[i][:,idx]           # applying to shuffle
        x_[i] = x_[i][:,idx]

        xyc_info.append([x[i], y[i], 'train-idx%d' % (i+1)])

    for i in range(noOfTask):
        xyc_info.append([x_[i], y_[i], 'test-idx%d' % (i+1)])

    return x, y, x_, y_, xyc_info
