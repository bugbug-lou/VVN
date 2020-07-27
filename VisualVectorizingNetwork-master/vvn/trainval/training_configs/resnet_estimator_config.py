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

IMAGENET_LEN = 1281167
BATCH_SIZE = 256
STEPS_PER_EPOCH = int(IMAGENET_LEN / BATCH_SIZE)

WEIGHT_DECAY = 1e-4

def loss_fn(outputs, inputs, **kwargs):
    logits = outputs['logits']
    labels = inputs['labels']
    return tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

def metrics_fn(outputs, inputs, **kwargs):
    logits = outputs['logits']
    labels = inputs['labels']
    preds = tf.argmax(logits, axis=1)

    return {
        'accuracy': tf.metrics.accuracy(labels, preds),
        'accuracy_top_5': tf.metrics.mean(
                            tf.nn.in_top_k(predictions=logits,
                                           targets=labels,
                                           k=5,
                                           name='top_5_op'))
    }

config = {
    'save_params': {
        'save_steps': STEPS_PER_EPOCH
    },
    'train_params': {
        'max_train_steps': 100 * STEPS_PER_EPOCH,
        'eval_steps': STEPS_PER_EPOCH,
        'batch_size': BATCH_SIZE,
    },
    'model_params': {
        'model_class': resnet_v1,
        'model_init_kwargs': {
            'resnet_depth': 18,
            'num_classes': 1000,
            'weight_decay': WEIGHT_DECAY
        }
    },
    'optimizer_params': {
        'optimizer_class': tf.train.MomentumOptimizer,
        'optimizer_init_kwargs': {
            'momentum': 0.9,
        }
    },
    'learning_rate_params': {
        'lr_func': tf.train.exponential_decay,
        'batch_denom': BATCH_SIZE,
        'lr_func_kwargs': {
            'learning_rate': 0.1,
            'decay_rate': 0.1,
            'decay_steps': STEPS_PER_EPOCH * 30,
            'staircase': True
        }
    },
    'train_data_params': {
        'data_class': ImageNet,
        'data_init_kwargs': {
            'prep_type': 'resnet',
            'crop_size': CROP_SIZE,
            'resize': RESIZE,
            'images_key': 'images',
            'labels_key': 'labels',
            'do_color_normalize': True
        }
    },
    'val_data_params': {
        'data_class': ImageNet,
        'data_init_kwargs': {
            'prep_type': 'resnet',
            'crop_size': CROP_SIZE,
            'resize': RESIZE,
            'images_key': 'images',
            'labels_key': 'labels',
            'do_color_normalize': True
        }
    },
    'loss_params': {
        'loss_func': loss_fn,
        'loss_func_kwargs': {}
    },
    'metrics_params': {
        'metrics_func': metrics_fn,
        'metrics_func_kwargs': {}
    },
}
