from dl4cv.layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU
  
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
  
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def affine_relu_norm_forward(x, w, b, gamma, beta, bn_param):
    ar_out, ar_cache = affine_relu_forward(x, w, b)
    norm_out, norm_cache = batchnorm_forward(ar_out, gamma, beta, bn_param)
    cache = (ar_cache, norm_cache)
    return norm_out, cache

def affine_relu_norm_backward(dout, cache):
    ar_cache, norm_cache = cache
    dnorm, dgamma, dbeta = batchnorm_backward(dout, norm_cache)
    dx, dw, db = affine_relu_backward(dnorm, ar_cache)
    return dx, dw, db, dgamma, dbeta

def affine_relu_dropout_norm_forward(x, w, b, gamma, beta, bn_param, dropout_param):
    ar_out, ar_cache = affine_relu_forward(x, w, b)
    dropout_out, dropout_cache = dropout_forward(ar_out, dropout_param)
    norm_out, norm_cache = batchnorm_forward(dropout_out, gamma, beta, bn_param)
    cache = (ar_cache, dropout_cache, norm_cache)
    return norm_out, cache

def affine_relu_dropout_norm_backward(dout, cache):
    ar_cache, dropout_cache, norm_cache = cache
    dnorm, dgamma, dbeta = batchnorm_backward(dout, norm_cache)
    ddropout = dropout_backward(dnorm, dropout_cache)
    dx, dw, db = affine_relu_backward(ddropout, ar_cache)
    return dx, dw, db, dgamma, dbeta