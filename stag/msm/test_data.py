import numpy as np


TRIMMABLE = {
    'assigns': np.array(
        [([0]*30 + [1]*20 + [-1]*10),
         ([2]*20 + [-1]*5 + [1]*35),
         ([0]*10 + [1]*30 + [2]*19 + [3])]),
    'no_trimming': {
        'implied_timescales': {
            'normalize': np.array(
                [[1.,  19.495726],
                 [2.,  19.615267],
                 [3.,  20.094898],
                 [4.,  19.796650]]),
            'transpose': np.array(
                [[1., 38.497835],
                 [2., 36.990989],
                 [3., 35.478863],
                 [4., 33.960748]])
            },
        },
    'trimming': {
        'implied_timescales': {
            'transpose': np.array(
                [[1., 25.562856],
                 [2., 24.384637],
                 [3., 23.198114],
                 [4., 22.001933]])
            },
        }
    }
