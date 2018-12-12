"""Get sliding windows from the real data. (Revised: 2018-09-03 00:30:01)

"""

import os
import shutil
from glob import glob

# !python download.py celebA

import numpy as np
import pandas as pd


# %% Column Scaling: Range ---------------------------------------------------

class RangeScaler(object):

    def __init__(
        self,
        real_range=(-10., 10.),
        feature_range=(0., 1.),
        copy=True,
        ):

        self.base_range = np.array((0., 1.))
        self.real_range = np.array(real_range, dtype=np.float32)
        self.feature_range = np.array(feature_range, dtype=np.float32)
        self.copy = copy

        self.real_min, self.real_max = real_range
        self.real_length = real_range[1] - real_range[0]

        self.feature_min, self.feature_max = feature_range
        self.feature_length = feature_range[1] - feature_range[0]
        self.feature_mean = np.mean(feature_range)


    def _check_array(self, input_x):
        if isinstance(input_x, np.ndarray):
            return input_x
        elif isinstance(input_x, (pd.Series, pd.DataFrame)):
            return input_x.values
        else:
            raise TypeError(
            "Input must be one of this: " +
            "['numpy.ndarray', 'pandas.Series', 'pandas.DataFrame']"
            )


    def _partial_scale(self, input_x):

        input_x = self._check_array(input_x)
        base_ranged = (
            (input_x - self.real_min) / self.real_length
        )

        if np.array_equal(self.feature_range, self.base_range):
            return base_ranged
        else:
            return (base_ranged * self.feature_length) + self.feature_min


    def transform(self, input_x):
        return self._partial_scale(input_x)


    def inverse_transform(self, scaled_x):
        base_ranged = (scaled_x - self.feature_min) / self.feature_length

        if np.array_equal(self.feature_range, self.base_range):
            return base_ranged
        else:
            return (base_ranged * self.real_length) + self.real_min


def column_range_scaler(
    dataframe,
    vendor_name,
    col_real_range_dict=None,
    feature_range=(-1., 1.),
    use_clip=False,
    ):

    if set(dataframe.columns) - set(col_real_range_dict):
        raise AttributeError(
            "'col_real_range_dict' should contains all columns in 'dataframe'"
        )
    else:
        result_frame = dataframe.copy()
        scaler_dict = dict()
        #for col, real_range in col_real_range_dict.items():
        for col in dataframe.columns:

            if col in ('SINR', 'RSRP', 'RSRQ'):
                col_real_range = col_real_range_dict[col][vendor_name]
            else:
                col_real_range = col_real_range_dict[col]

            scaler = RangeScaler(
                real_range=col_real_range,
                feature_range=feature_range,
            )
            # if result_frame[col].isnull().all():
            #     result_frame[col].fillna(
            #         np.mean(col_real_range),
            #         inplace=True,
            #     )
            scaled = scaler.transform(result_frame[[col]])

            scaler_dict[col] = scaler

            if result_frame[col].isnull().all():
                result_frame[col] = np.mean(feature_range)
            else:
                result_frame[col] = scaled

        return result_frame, scaler_dict


# full_scaled, full_scaler_dict = column_range_scaler(
#     daily_filled,
#     col_real_range_dict=col_range_dict,
#     feature_range=(-0.7, 1.),
# )



def load_data():

    data_str_list = [
        'train_x_mlp',
        'train_y_mlp',
        'train_x_rnn',
        'train_y_rnn',
        'test_x_mlp',
        'test_y_mlp',
        'test_x_rnn',
        'test_y_rnn',
    ]

    print('Loading Data...')
    data_dict = {}
    for _ in data_str_list:
        _tmp = np.load(f'data/{_}.npy').astype(np.float32)
        print(f'{_:10} : {_tmp.shape}')
        data_dict[_] = _tmp

    print('Complete.')
    
    return data_dict

