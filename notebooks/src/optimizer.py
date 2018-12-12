import numpy as np


def sgd(w, dw, param_dict=None):
    """
    - learning_rate: Scalar learning rate.
    """
    learning_rate = param_dict['learning_rate']

    
    #==== Essential ========================================================#
    
    new_w = w - (learning_rate * dw)
    
    #=======================================================================#
    
    
    return new_w, param_dict


def sgd_momentum(w, dw, param_dict=None):
    """
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a moving
      average of the gradients.
    """
    learning_rate = param_dict['learning_rate']
    momentum = param_dict['momentum'] = 0.9

    if 'velocity' not in param_dict:
        param_dict['velocity'] = np.zeros_like(w)
    velocity = param_dict['velocity']

    
    #==== Essential ========================================================#
    
    v = (momentum * velocity) - (learning_rate * dw)
    new_w = w + v
    param_dict['velocity'] = v
    
    #=======================================================================#
    
    
    return new_w, param_dict


def adagrad(w, dw, param_dict=None):
    """
    - learning_rate: Scalar learning rate.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    learning_rate = param_dict['learning_rate']
    decay_rate = param_dict['decay_rate'] = 0.99
    epsilon = param_dict['epsilon'] = 1e-8
    
    if 'cache' not in param_dict:
        param_dict['cache'] = np.zeros_like(w)
    cache = param_dict['cache']
    
    
    #==== Essential ========================================================#
    
    cache  = param_dict['cache'] = cache + (dw**2)
    new_w = w - ((learning_rate * dw) / (np.sqrt(cache) + epsilon))
    param_dict['cache'] = cache
    
    #=======================================================================#
    
    
    return new_w, param_dict


def rmsprop(w, dw, param_dict=None):
    """
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    learning_rate = param_dict['learning_rate']
    decay_rate = param_dict['decay_rate'] = 0.9
    epsilon = param_dict['epsilon'] = 1e-8
    
    if 'cache' not in param_dict:
        param_dict['cache'] = np.zeros_like(w)
    cache = param_dict['cache']
    
    
    #==== Essential ========================================================#
    
    cache = param_dict['cache'] = (decay_rate * cache) + ((1 - decay_rate) * (dw**2))
    new_w = w - ((learning_rate * dw) / (np.sqrt(cache) + epsilon))
    
    #=======================================================================#
    
    
    return new_w, param_dict


def adam(w, dw, param_dict=None):
    """
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    learning_rate = param_dict['learning_rate']
    beta1 = param_dict['beta1'] = 0.9
    beta2 = param_dict['beta2'] = 0.999
    epsilon = param_dict['epsilon'] = 1e-8
    
    if 'm' not in param_dict:
        param_dict['m'] = np.zeros_like(w)
    
    if 'v' not in param_dict:
        param_dict['v'] = np.zeros_like(w)

    if 't' not in param_dict:
        param_dict['t'] = 0
    
    m = param_dict['m']
    v = param_dict['v']
    t = param_dict['t']
    
    
    #==== Essential ========================================================#
    
    learning_rate = learning_rate * (np.sqrt(1 - beta2) / (1 - beta1))
    t = param_dict['t'] = 1
    m = param_dict['m'] = (beta1 * m) + ((1 - beta1) * dw)
    v = param_dict['v'] = (beta2 * v) + ((1 - beta2) * (dw**2))
    
    mt = m / (1 - (beta1 ** t))
    mv = v / (1 - (beta2 ** t))
    
    new_w = w - ((learning_rate * mt) / (np.sqrt(mv) + epsilon))
    param_dict['learning_rate'] = learning_rate
    
    #=======================================================================#
    
    return new_w, param_dict


