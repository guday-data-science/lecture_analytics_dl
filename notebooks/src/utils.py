import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint


def qprint(arr_like):
    print(arr_like.shape)
    pprint(arr_like)

    
def lprint(input_x, output, name=None):
    print(f'{str(name)}:\t{input_x.shape} -> {output.shape}')

    
def keras_lossplot(keras_model, figsize=(14, 7)):
    
    fitted = keras_model.history

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(fitted.history['loss'], label='train')
    if 'val_loss' in fitted.history:
        ax.plot(fitted.history['val_loss'], label='valid')
    ax.legend()
    
    plt.close(fig)
    return fig


def keras_predict_plot(model, x, y, method='rnn'):
    
    if method == 'mlp':
        
        y_pred = model.predict(x)
        y_real = y
    
    if method == 'rnn':
        
        y_pred = model.predict(x)[:, 1]
        y_real = y[:, 1]

    kpi_num = y_real.shape[-1]
    fig, axes = plt.subplots(
        nrows=kpi_num,
        ncols=1,
        figsize=(18, 4 * kpi_num),
    )

    for i in range(kpi_num):
        ax = axes[i]
        ax.plot(y_pred[:, i], label='pred')
        ax.plot(y_real[:, i], label='real')
        ax.legend()

    fig.show()

    
def tf_predict_plot(y_real, y_pred, method='rnn'):
    
    if method == 'mlp':
        
        y_real = y_real
        y_pred = y_pred
    
    if method == 'rnn':
        
        y_real = y_real[:, 1]
        y_pred = y_pred[:, 1]

    kpi_num = y_real.shape[-1]
    fig, axes = plt.subplots(
        nrows=kpi_num,
        ncols=1,
        figsize=(18, 4 * kpi_num),
    )

    for i in range(kpi_num):
        ax = axes[i]
        ax.plot(y_pred[:, i], label='pred')
        ax.plot(y_real[:, i], label='real')
        ax.legend()

    fig.show()

