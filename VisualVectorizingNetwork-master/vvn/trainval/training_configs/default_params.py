import tfutils
import tensorflow.compat.v1 as tf

DEFAULT_TFUTILS_PARAMS = {
    'train_params': {
        'queue_params': None,
        'thres_loss': float('inf'),
        'num_steps': float('inf'),  # number of steps to train
    },

    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'decay_rate': 1.0, # constant learning rate
        'decay_steps': 1e6
    },

    'optimizer_params': {
        'optimizer': tfutils.optimizer.ClipOptimizer,
        'optimizer_class': tf.train.AdamOptimizer,
        'clip': True,
        'clipping_method': 'norm',
        'clipping_value': 10000.0
    },

    'log_device_placement': False,  # if variable placement has to be logged
}
