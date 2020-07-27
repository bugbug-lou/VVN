import tensorflow.compat.v1 as tf
import vvn.models.losses as loss_functions
from vvn.ops.utils import preproc_hsv
from vvn.models.configs.graph_op_configs import collide_top_ops, push_forward_top_ops

NUM_GPUS = 1
DEVICES = ['/gpu:' + str(i) for i in range(NUM_GPUS)]

TOTAL_BATCH_SIZE = 4
MB_SIZE = 4

POSTFIX = ''
IMAGES = 'images' + POSTFIX
DEPTHS = 'depths' + POSTFIX
NORMALS = 'normals' + POSTFIX
OBJECTS = 'objects' + POSTFIX
INPUT_SIZE = 256
RESIZES = {k:[INPUT_SIZE]*2 for k in [IMAGES, DEPTHS, NORMALS, OBJECTS]}

INPUT_SEQUENCE_LEN = 5
NUM_LEVELS = None

NUM_TRAIN_EXAMPLES = 102400
NUM_VAL_EXAMPLES = NUM_TRAIN_EXAMPLES // 9

# because some of the metas are labeled wrong
sfx = '64'
RESIZES.update({k+sfx:v for k,v in RESIZES.items()})

encoder_params_train = {
    # temporal
    'inp_sequence_len': INPUT_SEQUENCE_LEN,
    'output_times': [0],

    # input names
    'images_key': IMAGES,
    'depths_key': DEPTHS,
    'normals_key': NORMALS,
    'objects_mask_key': OBJECTS,
    'segments_key': 'segments',

    # which inputs to use
    'hsv_input': True,
    'xy_input': True,
    'depths_input': True,
    'normals_input': True,
    'hw_input': False,
    'ones_input': True,
    'objects_mask_input': True,
    'diff_x_input': False,
    'diff_t_input': True,

    # input preprocessing params
    'color_normalize': False,
    'color_scale': 255.0,
    'depth_normalization': 100.1, # new tfrecords
    'background_depth': 30.0,
    'negative_depths': True,
    'near_plane': 0.1,

    # vectorizing params
    'aggregation_kwargs': {'agg_vars': True},
    'add_spatial_attributes': True,
    'add_border_attributes': True,
    'add_previous_attributes': ['positions', 'positions_var', 'normals', 'normals_var', 'positions_backward_euler', 'normals_backward_euler', 'hw_centroids', 'areas'],
    'history_times': 3,
    'mlp_kwargs': {'num_features':[250,250,15+64]},

    # architecture
    'stem_model_func': None,
    'num_levels': NUM_LEVELS
}

dynamics_model_params = {
    'graph_construction_kwargs': {
        'model_name': None,
        'build_object_nodes':False,
        'stop_all_gradients': False,
        'attribute_dims': [{'velocities':'positions_backward_euler'},{'velocities':[-40,-37], 'shapes': [-37,-5]}],
        # 'attribute_dims': [{'velocities':'positions_backward_euler'},
        #                    {'velocities':'positions_backward_euler', 'shapes': [-37,-5]}],
        'attribute_preprocs': {'normals': lambda n: tf.nn.l2_normalize(n, axis=-1), 'hsv':preproc_hsv}
    },
    'graph_ops_list': collide_top_ops + push_forward_top_ops,
    # 'graph_ops_list': push_forward_top_ops,
    # 'graph_ops_list': [],
}

prediction_kwargs = {
    'dynamics_loss_func': loss_functions.node_attr_mse,
    'dynamics_loss_func_kwargs': {'loss_attrs':['positions'], 'valid_attr':'inview'},
    'max_rollout_times': 1
}

tdw_data_params = {
    'dataset_names': ['playroom_v1'],
    'dataprefix': '/data4/dbear/tdw_datasets/',
    'sources': [IMAGES, DEPTHS, NORMALS,
                OBJECTS, 'projection_matrix', 'camera_matrix'],
    'num_train_examples': NUM_TRAIN_EXAMPLES,
    'num_val_examples': NUM_VAL_EXAMPLES,
    'train_filter_rule': None,
    'val_filter_rule': None,
    'enqueue_batch_size': 10,
    'delta_time':1,
    'buffer_mult':10,
    'motion_filter': False,
    'motion_thresh': 0.03,
    'motion_area_thresh': 0.1,
    'resizes': RESIZES,
    'get_segments': True
}

config = {
    'train_params': {
        'batch_size': TOTAL_BATCH_SIZE,
        'minibatch_size': MB_SIZE,
        'queue_params': None,
        'thres_loss': float('inf'),
        'num_steps': 1000000,
        'save_params': {
            'do_save': True,
            'save_initial_filters': True,
            'save_metrics_freq': 100,
            'host': 'localhost',
            'dbname': 'dyn',
            'collname': 'gtinp',
            'exp_id': 'seq5del1_playV1_1',
            'port': 27024,
            'save_valid_freq': 10000,
            'save_filters_freq': 500,
            'cache_filters_freq': 10000
        },
        'load_params': {
            'dbname': 'dyn',
            'collname': 'gtinp',
            'exp_id': 'seq5del1_playV1_1',
        },
        'model_params': {
            'train_targets': [IMAGES, DEPTHS, NORMALS,
                              OBJECTS, 'projection_matrix', 'camera_matrix'],
            'visible_times': INPUT_SEQUENCE_LEN,
            'num_levels': NUM_LEVELS,
            'dynamics_targets': ['projection_matrix'],
            'prediction_kwargs': prediction_kwargs,
            'dynamics_loss_func': prediction_kwargs.get('dynamics_loss_func', None),
            'dynamics_loss_func_kwargs': prediction_kwargs.get('dynamics_loss_func_kwargs', {}),
            'devices': DEVICES,
            'num_gpus': NUM_GPUS
        }
    },

    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'decay_rate': 1.0, # constant learning rate
        'decay_steps': 1e6
    },

    'optimizer_params': {
        'optimizer': tf.train.AdamOptimizer,
        'optimizer_kwargs': {
            'learning_rate': 2e-4,
        },
        'clip': True,
        'clipping_norm': 10000.0
    },

    'loss_params': {
        'loss_func': loss_functions.end_to_end_loss,
        'loss_func_kwargs': {
            'dynamics_loss_scale': 1000.0
        }
    },

    'encoder_params': encoder_params_train,

    'dynamics_params': dynamics_model_params,

    'prediction_kwargs': prediction_kwargs,

    'data_params': tdw_data_params,

    'log_device_placement': False,
}