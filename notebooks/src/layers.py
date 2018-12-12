import numpy as np
from .utils import qprint, lprint
from unipy import aprint, lprint


def mean_squared_error(logits, real):
    
    
    #==== Essential ========================================================#
    
    err = ((logits - real) ** 2).mean()
    
    #=======================================================================#

    
    return err


def sigmoid(x, print_ok=True):


    res = 1 / (1 + np.exp(-x))

    if print_ok:
        print('sigmoid')
        aprint(x, res, name_list=['Raw', 'Activated'])
    return res


def tanh(x):
    return np.tanh(x)


def d_sigmoid(x):
        return sigmoid(x, print_ok=False) * (1 - sigmoid(x, print_ok=False))

    
def d_tanh(x):
    return 1 - (tanh(x) ** 2)


def dense_layer(
    input_x,
    output_dim=None,
    weight=None,
    bias=None,
    name=None,
    seed=1,
    print_ok=True,
    ):
    
    input_dim = input_x.shape[-1]

    
    #==== Essential ========================================================#

    if weight is None:
        np.random.seed(seed)
        weight = np.random.random((input_dim, output_dim)).round(2)
        bias = np.random.random((output_dim,)).round(2)

    output = (input_x @ weight) + bias

    #=======================================================================#

    
    if print_ok:
        lprint(input_x, output, name=name)

    return output, weight, bias
