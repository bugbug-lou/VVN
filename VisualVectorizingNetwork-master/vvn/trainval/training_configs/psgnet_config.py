import tensorflow.compat.v1 as tf
from tfutils import optimizer
import vvn.models as models
import vvn.ops as ops
from vvn.trainval.utils import collect_and_flatten, total_loss
from vvn.data.tdw_data import TdwSequenceDataProvider

MODEL_PREFIX = 'model_0'
POSTFIX = ''
IMAGES = 'images' + POSTFIX
DEPTHS = 'depths' + POSTFIX
NORMALS = 'normals' + POSTFIX
OBJECTS = 'objects' + POSTFIX
INPUT_SIZE = 256
RESIZES = {k:[INPUT_SIZE]*2 for k in [IMAGES, DEPTHS, NORMALS, OBJECTS]}

INPUT_SEQUENCE_LEN = 4

config = {
    'model_params': {
        'preprocessor': {
            'model_func': models.preprocessing.preproc_tensors_by_name,
            'dimension_order': ['images'],
            'dimension_preprocs': {'images': models.preprocessing.preproc_rgb}
        },
        'extractor': {
            'model_func': ops.convolutional.convnet_stem,
            'name': 'ConvNet',
            'ksize': 7, 'max_pool': True, 'conv_kwargs': {'activation': 'relu'}
        },
        'decoders': [
            {'name': 'avg_pool', 'model_func': ops.convolutional.global_pool, 'kind': 'avg', 'keep_dims': True},
            {'name': 'classifier', 'model_func': ops.convolutional.fc, 'out_depth': 1001}
        ],
        'losses': [
            {'name': 'L2', 'loss_func': models.losses.l2_loss, 'scale':1.0}
        ],
        'inp_sequence_len': INPUT_SEQUENCE_LEN,
        'to_decode': None,
        'losses_now': [
            {'name': 'L2', 'logits_mapping': {'logits': 'avg_pool/outputs'}, 'labels_mapping': {'labels': 'avg_pool/outputs'}}
        ],
        'train_targets': [IMAGES, DEPTHS, NORMALS, OBJECTS]
    },
    'data_params': {
        'func': TdwSequenceDataProvider,
        'delta_time': 1,
        'enqueue_batch_size': 10,
        'buffer_mult': 10,
        'motion_filter': False,
        'motion_thresh': 0.03,
        'motion_area_thresh': 0.1,
        'train_filter_rule': None,
        'val_filter_rule': None,
        'resizes': RESIZES,
        'sources': [IMAGES, DEPTHS, NORMALS, OBJECTS],
        'n_tr_per_dataset': 102400,
        'n_val_per_dataset': 10240
    },
    'loss_params': {
        'pred_targets': [IMAGES, DEPTHS, NORMALS, OBJECTS],
        'loss_func': total_loss,
        'loss_func_kwargs': {}
    }
}
