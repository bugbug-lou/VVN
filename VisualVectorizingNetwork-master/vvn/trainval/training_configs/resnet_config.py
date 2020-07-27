import tensorflow.compat.v1 as tf
from tfutils import optimizer
import vvn.models as models
from vvn.models.resnets.resnet_model import resnet_v1
import vvn.ops as ops
import vvn.trainval.eval_metrics as eval_metrics
from vvn.trainval.utils import collect_and_flatten, total_loss
from vvn.data.imagenet_data import ImageNet

MODEL_PREFIX = 'model_0'
CROP_SIZE = 256
RESIZE = 256
INPUT_SEQUENCE_LEN = 1

WEIGHT_DECAY = 1e-4
RESNET18 = resnet_v1(18, num_classes=None, weight_decay=WEIGHT_DECAY)

config = {
    'save_params': {
        'save_valid_freq': 5004,
        'save_filters_freq': 5004,
        'cache_filters_freq': 5004
    },
    'model_params': {
        'preprocessor': {
            'model_func': models.preprocessing.preproc_tensors_by_name,
            'dimension_order': ['images'],
            'dimension_preprocs': {'images': models.preprocessing.preproc_rgb}
        },
        'extractor': {
            'model_func': RESNET18, 'base_tensor_name': 'block0',
            'name': 'ResNet18', 'layer_names': ['pool']
        },
        'decoders': [
            {'name': 'classifier', 'model_func': ops.convolutional.fc, 'out_depth': 1000, 'kernel_init': 'random_normal', 'kernel_init_kwargs':{'stddev': 0.01, 'seed':0}, 'weight_decay': WEIGHT_DECAY, 'input_mapping': {'inputs': 'features/pool'}}
        ],
        'losses': [
            {
                'name': 'CE', 'required_decoders': ['classifier'], 'loss_func': models.losses.sparse_ce, 'scale':1.0,
                'logits_mapping': {'logits': 'classifier/outputs'}, 'labels_mapping': {'labels': 'labels'}
            }
        ],
        'inp_sequence_len': INPUT_SEQUENCE_LEN,
        'train_targets': ['labels']
    },
    'optimizer_params': {
        'optimizer': optimizer.ClipOptimizer,
        'optimizer_class': tf.train.MomentumOptimizer,
        'clip': False,
        'momentum': 0.9,
        #'optimizer_kwargs': {
        #}
    },
    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'learning_rate': 0.1,
        'decay_rate': 0.1,
        'decay_steps': 5004 * 30,
        'staircase': True
    },
    'data_params': {
        'func': ImageNet,
        'prep_type': 'resnet',
        'crop_size': CROP_SIZE,
        'resize': RESIZE,
        'images_key': 'images',
        'labels_key': 'labels',
        'do_color_normalize': False
    },
    'loss_params': {
        'pred_targets': ['labels'],
        'loss_func': total_loss,
        'loss_func_kwargs': {}
    },
    'validation_params': {
        'accuracy': {
            'targets': {
                'func': eval_metrics.loss_and_in_top_k,
                'target': 'labels',
                'logits_key': 'classifier/outputs'
            },
            'val_length': ImageNet.VAL_LEN,
            'online_agg_func': eval_metrics.online_agg,
            'agg_func': eval_metrics.mean_res

        }
    }
}
