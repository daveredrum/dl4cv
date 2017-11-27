import pickle as pickle
import numpy as np
import os


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        # load with encoding because file was pickled with Python 2
        datadict = pickle.load(f, encoding='latin1')
        X = np.array(datadict['data'])
        Y = np.array(datadict['labels'])
        X = X.reshape(-1, 3, 32, 32).transpose(0,2,3,1).astype("float")
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    f = os.path.join(ROOT, 'cifar10_train.p')
    Xtr, Ytr = load_CIFAR_batch(f)
    return Xtr, Ytr


def scoring_function(x, lin_exp_boundary, doubling_rate):
    assert np.all([x >= 0, x <= 1])
    score = np.zeros(x.shape)
    lin_exp_boundary = lin_exp_boundary
    linear_region = np.logical_and(x > 0.1, x <= lin_exp_boundary)
    exp_region = np.logical_and(x > lin_exp_boundary, x <= 1)
    score[linear_region] = 100.0 * x[linear_region]
    c = doubling_rate
    a = 100.0 * lin_exp_boundary / np.exp(lin_exp_boundary * np.log(2) / c)
    b = np.log(2.0) / c
    score[exp_region] = a * np.exp(b * x[exp_region])
    return score