import numpy as np

from ..msm.transition_matrices import TrimMapping


TRIMMABLE = {
    'assigns': np.array(
        [([0]*30 + [1]*20 + [-1]*10),
         ([2]*20 + [-1]*5 + [1]*35),
         ([0]*10 + [1]*30 + [2]*19 + [3])]),
    'no_trimming': {
        'msm': {
            'normalize': {
                'tcounts_': np.array([[38,  2,  0, 0],
                                      [ 0, 82,  1, 0],
                                      [ 0,  1, 37, 1],
                                      [ 0,  0,  0, 0]]),
                'tprobs_': np.array([[0.95, 0.05    , 0.      , 0.      ],
                                     [0.  , 0.987951, 0.012048, 0.      ],
                                     [0.  , 0.025641, 0.948717, 0.025641],
                                     [0.  , 0.      , 0.      , 0.      ]]),
                'eq_probs_': np.array([0., 0.788068, 0.206606, 0.005326]),
                'mapping_': TrimMapping([(0, 0), (1, 1), (2, 2), (3, 3)])
            },
            'transpose': {
                'tcounts_': np.array([[38,  1,   0,   0],
                                      [ 1, 82,   1,   0],
                                      [ 0,  1,  37, 0.5],
                                      [ 0,  0, 0.5,   0]]),
                'tprobs_': np.array([[0.974358, 0.025641, 0.      , 0.     ],
                                     [0.011904, 0.976190, 0.011905, 0.     ],
                                     [0.      , 0.025974, 0.961038, 0.01299],
                                     [0.      , 0.      , 1.      , 0.     ]]),
                'eq_probs_': np.array([0.240741, 0.518519, 0.237654, .003086]),
                'mapping_': TrimMapping([(0, 0), (1, 1), (2, 2), (3, 3)])
            }
        },
        'implied_timescales': {
            'normalize': np.array(
                [[19.495726],
                 [19.615267],
                 [20.094898],
                 [19.796650]]),
            'transpose': np.array(
                [[38.497835],
                 [36.990989],
                 [35.478863],
                 [33.960748]])
            },
        },
    'trimming': {
        'msm': {
            'normalize': {
                'tcounts_': np.array([[82,  1],
                                      [ 1, 37]]),
                'tprobs_': np.array([[ 0.987952,  0.012048],
                                     [ 0.026316,  0.973684]]),
                'eq_probs_': np.array([ 0.68595,  0.31405]),
                'mapping_': TrimMapping([(1, 0), (2, 1)])
            },
            'transpose': {
                'tcounts_': np.array([[82,  1],
                                      [ 1, 37]]),
                'tprobs_': np.array([[ 0.987952,  0.012048],
                                     [ 0.026316,  0.973684]]),
                'eq_probs_': np.array([ 0.68595,  0.31405]),
                'mapping_': TrimMapping([(1, 0), (2, 1)])
            }
        },
        'implied_timescales': {
            'transpose': np.array(
                [[25.562856],
                 [24.384637],
                 [23.198114],
                 [22.001933]])
            },
        }
    }
