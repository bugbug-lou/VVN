from __future__ import division, print_function, absolute_import
import os
import sys
import argparse

import numpy as np
import tensorflow as tf
import copy
from collections import OrderedDict

# tfutils is the package we use to run and log experiments
import tfutils.optimizer
import tfutils.base
import tfutils.defaults

# model functions
from vvn.models.visual_encoder import vectorize_inputs_model, collect_and_flatten
from vvn.models.dynamics import PhysicsModel, VisualPhysicsModel, DimensionDict
from vvn.models.configs.graph_op_configs import *
import vvn.models.losses as loss_functions

# for evaluating models
import vvn.trainval.eval_metrics as eval_metrics

# for getting TDW data into the models
from vvn.data.tdw_data import TdwSequenceDataProvider
from vvn.data.data_utils import *

import pickle
import dill
import pdb
import logging

train_net=True
if train_net:
    TOTAL_BATCH_SIZE = 4
    MB_SIZE = 4
    NUM_GPUS = 1
    VAL_BATCH_SIZE= 4
else:
    TOTAL_BATCH_SIZE = 8
    MB_SIZE = 8
    NUM_GPUS = 1
    VAL_BATCH_SIZE= 8

if not isinstance(NUM_GPUS, list):
    DEVICES = ['/gpu:' + str(i) for i in range(NUM_GPUS)]
else:
    DEVICES = ['/gpu:' + str(i) for i in range(len(NUM_GPUS))]

# data params
NUM_TRAIN_EXAMPLES = 102400
NUM_VAL_EXAMPLES = NUM_TRAIN_EXAMPLES // 9

MODEL_PREFIX = 'model_0'
POSTFIX = ''
IMAGES = 'images' + POSTFIX
DEPTHS = 'depths' + POSTFIX
NORMALS = 'normals' + POSTFIX
OBJECTS = 'objects' + POSTFIX
STANDARD_TRANS_DICT = {
    IMAGES: 'images', DEPTHS: 'depths', NORMALS: 'normals', OBJECTS: 'objects', 'camera':'camera', 'valid':'valid', 'projection_matrix':'projection_matrix', 'camera_matrix': 'camera_matrix', 'full_particles_agent': 'full_particles_agent', 'full_particles_agent_mask': 'full_particles_agent_mask'
}

INPUT_SIZE = 256
RESIZES = {k:[INPUT_SIZE]*2 for k in [IMAGES, DEPTHS, NORMALS, OBJECTS]}

# because some of the metas are labeled wrong
sfx = '64'
RESIZES.update({k+sfx:v for k,v in RESIZES.items()})

DEFAULT_INTEGRATED_PARAMS = {
    'train_params': {
        'minibatch_size': MB_SIZE,
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

def end_to_end_loss(
        logits,
        labels,
        dynamics_loss_scale = 1.0,
        **kwargs
):
    """
    Wrapper to combine particle encoder loss with hrn loss.
    """
    loss = 0.0
    # dynamics loss
    if 'dynamics_loss' in logits and dynamics_loss_scale > 0.0:
        print("Using Dynamics loss scaled by %f" % dynamics_loss_scale)
        loss += logits['dynamics_loss']['dynamics_loss'] * dynamics_loss_scale

    return loss


def train_vectorizing_and_dynamics_model(default_params,
                                         dbname='integ', collname='test', exp_id='exp0', port=27021,
                                         save_valid_freq=5000, save_filters_freq=5000, validate_first=False,
                                         load=False, load_port=None, load_exp_id=None, load_step=None, restore_global_step=True,
                                         dataprefix='/mnt/fs1/datasets/', dataset_names=['easy/collide10'],
                                         train_targets_list=[IMAGES, DEPTHS, NORMALS, OBJECTS, 'projection_matrix', 'camera_matrix'],
                                         camera_normals=True,
                                         sequence_len=None,
                                         train_on_tdw=False,
                                         val_batch_size=None,
                                         data_params={'crop_size':256, 'resize':None},
                                         model_params={'inp_sequence_len':4, 'ntimes':12, 'time_dilation':3},
                                         dynamics_model=None,
                                         dynamics_model_params={},
                                         prediction_kwargs={},
                                         loss_func=None,
                                         loss_func_kwargs={},
                                         validate_node_attributes=True,
                                         validation_func=None,
                                         validation_kwargs=None,
                                         validate_object_metrics=False,
                                         validate_object_metrics_kwargs={'bg_keys':[], 'compute_BDEs':0, 'compute_ARIs':0},
                                         learning_rate=None,
                                         lr_params=None,
                                         opt_params=None,
                                         train_encoder=True,
                                         trainable_scope=None,
                                         cfg_name=None
):
    bs = TOTAL_BATCH_SIZE
    T = model_params['inp_sequence_len']
    L = model_params.get('num_levels', None)
    if sequence_len is None:
        sequence_len = T

    train_params = copy.deepcopy(default_params)
    train_params['save_params'] = {
        'do_save': True,
        'save_initial_filters': True,
        'save_metrics_freq': 500,
        'host': 'localhost',
        'dbname': dbname,
        'collname': collname,
        'exp_id': exp_id,
        'port': port,
        'save_valid_freq': save_valid_freq,
        'save_filters_freq': save_filters_freq,
        'cache_filters_freq': save_filters_freq
    }

    ### LOADING ###
    if load:
        train_params['load_params'] = {
            'host': 'localhost',
            'port': load_port if load_port is not None else port,
            'dbname': dbname,
            'collname': collname,
            'exp_id': load_exp_id or exp_id,
            'do_restore': True,
            'query': {'step':load_step} if load_step is not None else None,
            'restore_global_step': restore_global_step
        }

    # set train params targets
    train_params_targets = []
    if prediction_kwargs.get('dynamics_loss_func', None) is not None:
        train_params_targets.append('dynamics_loss')

    ### INPUT DATA ###
    if data_params.get('sources', None) is None:
        data_params['sources'] = [t for t in train_targets_list if '_mask' not in t]
    train_data_params, val_data_params = TdwSequenceDataProvider.get_data_params(
        dataset_names=dataset_names,
        batch_size=bs,
        sequence_len=sequence_len,
        dataprefix=dataprefix,
        n_tr_per_dataset=NUM_TRAIN_EXAMPLES,
        n_val_per_dataset=NUM_VAL_EXAMPLES,
        **data_params)

    train_data_params['batch_size'] = bs
    val_data_params['batch_size'] = val_batch_size or bs
    def _data_input_fn_wrapper(batch_size, train=train_net, **kwargs):
        data_provider_cls = TdwSequenceDataProvider(**kwargs)
        return data_provider_cls.input_fn(batch_size, train)
    train_data_params['func'] = _data_input_fn_wrapper
    val_data_params['func'] = _data_input_fn_wrapper

    train_params_targets = {
        'func': collect_and_flatten,
        'targets': train_params_targets
    }

    train_params['train_params'].update({
        'minibatch_size': MB_SIZE,
        'validate_first': validate_first,
        'data_params': train_data_params,
        'targets': train_params_targets
    })

    ### ONLINE VALIDATION ###
    train_params['validation_params'] = {}
    if validate_node_attributes:
        train_params['validation_params']['valid0'] = {
            'data_params': val_data_params,
            'num_steps': NUM_VAL_EXAMPLES // val_data_params['batch_size'],
            'targets': {'func': validation_func,
                        'target': train_targets_list
            },
            'online_agg_func': eval_metrics.online_agg,
            'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()}
        }
        if validation_kwargs is None:
            validation_kwargs = {'loss_func_kwargs': loss_func_kwargs}
        for k,v in validation_kwargs.items():
            train_params['validation_params']['valid0']['targets'][k] = v

    if validate_object_metrics:
        validate_object_metrics_kwargs['take_every'] = len(model_params['output_times'])
        train_params['validation_params']['object_metrics'] = {
            'data_params': val_data_params,
            'num_steps': NUM_VAL_EXAMPLES // val_data_params['batch_size'],
            'targets': {'func': eval_metrics.get_pred_and_gt_segments,
                        'target': train_targets_list
            },
            'online_agg_func': eval_metrics.object_mask_and_boundary_metrics,
            'agg_func': eval_metrics.agg_mean_per_time
        }
        train_params['validation_params']['object_metrics']['targets'].update(copy.deepcopy(validate_object_metrics_kwargs))

    train_params['loss_params'] = {
        'labels_to_dict': True,
        'pred_targets': train_targets_list,
        'loss_func': loss_func,
        'loss_func_kwargs': loss_func_kwargs,
        'agg_func': tfutils.defaults.mean_and_reg_loss
    }

    ### MODEL ###
    encoder_model_params = copy.deepcopy(model_params)
    dynamics_model_params = copy.deepcopy(dynamics_model_params)
    model_func = VisualPhysicsModel(
        encoder_model_func=vectorize_inputs_model,
        encoder_model_params=encoder_model_params,
        physics_model_class=dynamics_model,
        physics_model_params=dynamics_model_params
    )
    train_params['model_params'] = {
        'func': model_func,
        'train_targets': train_targets_list,
        'visible_times': T,
        'num_levels': L,
        'dynamics_targets': ['projection_matrix'],
        'prediction_kwargs': prediction_kwargs,
        'dynamics_loss_func': prediction_kwargs.get('dynamics_loss_func', None),
        'dynamics_loss_func_kwargs': prediction_kwargs.get('dynamics_loss_func_kwargs', {}),
        'devices': DEVICES,
        'num_gpus': NUM_GPUS
    }

    ### OPTIMIZATION ###
    if lr_params is not None:
        train_params['learning_rate_params'] = lr_params
    elif learning_rate is not None:
        train_params['learning_rate_params']['learning_rate'] = learning_rate
    if opt_params is not None:
        train_params['optimizer_params'] = opt_params

    # train some variables only (e.g. not the pretrained ff weights)
    if trainable_scope is not None:
        train_params['optimizer_params']['trainable_scope'] = trainable_scope

    pkl_name = os.path.join('.', 'training_configs', exp_id if cfg_name is None else cfg_name)
    with open(pkl_name + '.pkl', 'wb') as f:
        dill.dump(train_params, f)

    tfutils.base.train_from_params(**train_params)

def train_from_cfg(cfg_name, dbname, collname, exp_id, port):

    cfg_path = os.path.join('~/images2particles/training_scripts/encoder_pickles', cfg_name)
    with open(cfg_path + '.pkl', 'rb') as f:
        train_params = dill.load(f)

    save_params = train_params.pop('save_params', {})
    save_params.update({
        'dbname': dbname,
        'collname': collname,
        'exp_id': exp_id,
        'port': port
    })
    train_params['save_params'] = save_params
    print(train_params)

def store_input_images(inputs, outputs, target, labels_trans_dict=STANDARD_TRANS_DICT, read_depths=True, **kwargs):

    if labels_trans_dict is None:
        labels_trans_dict = {k:k for k in target}

    print(target)
    print(labels_trans_dict)

    results = {labels_trans_dict[k]: inputs[k] for k in labels_trans_dict.keys() if k in target}
    print([r.shape for k,r in results.items()])

    if read_depths and 'depths' in results.keys():
        results['depths'] = read_depths_image(inputs['depths'], normalization=100.1)

    return results

def store_outputs(inputs, outputs, target, store_labels=True, labels_trans_dict=STANDARD_TRANS_DICT,
                        target_layers=['conv1'],
                        **kwargs):

    if labels_trans_dict is None:
        labels_trans_dict = {k:k for k in target}

    results = {labels_trans_dict[k]: inputs[k] for k in labels_trans_dict.keys() if k in target}
    print([(k,r.shape.as_list()) for k,r in results.items()])

    if not store_labels:
        results = {}

    for layer in target_layers:
        results[layer] = outputs[layer]
        print(layer, results[layer].shape)

    return results

def validate_integrated_model(
        dbname, collname, exp_id, port, load_step, dataset_names=['easy/collide10'], val_on_tdw=True,
        load_port=None, save_id=None, group='val', prefix='ev0', batch_size=TOTAL_BATCH_SIZE,
        num_steps=1, devices=DEVICES, num_gpus=NUM_GPUS, model_prefix=MODEL_PREFIX,
        target_func=store_outputs,
        target_func_kwargs={},
        online_agg_func=eval_metrics.append_it,
        agg_func=eval_metrics.just_keep_everything,
        val_targets_list=[IMAGES, DEPTHS, NORMALS, OBJECTS, 'projection_matrix', 'camera_matrix', 'full_particles_agent'] + ['reference_ids'],
        camera_normals=True,
        dataprefix='/mnt/fs1/datasets/',
        save_to_gfs=['nodes_level_0', 'nodes_level_1', 'nodes_level_0_pred', 'nodes_level_1_pred', 'segments_level_1', 'num_segments_level_1', 'dynamics_loss'],
        sequence_len=4,
        model_func=vectorize_inputs_model,
        model_params={},
        dynamics_model=PhysicsModel,
        dynamics_model_params={},
        prediction_kwargs={},
        data_params={},
        shuffle_val=False,
        shuffle_seed=0,
        pickle_results=False,
        local_pickle_dir='encoder_pickles',
        **kwargs
):
    params = {}
    encoder_model_params = copy.deepcopy(model_params)
    dynamics_model_params = copy.deepcopy(dynamics_model_params)
    model_func = VisualPhysicsModel(
        encoder_model_func=vectorize_inputs_model,
        encoder_model_params=encoder_model_params,
        physics_model_class=dynamics_model,
        physics_model_params=dynamics_model_params
    )
    params['model_params'] = {
        'func': model_func,
        'train_targets': val_targets_list,
        'visible_times': encoder_model_params['inp_sequence_len'],
        'num_levels': encoder_model_params.get('num_levels', None),
        'dynamics_targets': ['projection_matrix'],
        'prediction_kwargs': prediction_kwargs,
        'dynamics_loss_func': prediction_kwargs.get('dynamics_loss_func', None),
        'dynamics_loss_func_kwargs': prediction_kwargs.get('dynamics_loss_func_kwargs', {}),
        'devices': DEVICES,
        'num_gpus': NUM_GPUS
    }

    params['load_params'] = {
        'host': 'localhost',
        'port': load_port or port,
        'dbname': dbname,
        'collname': collname,
        'exp_id': exp_id,
        'do_restore': True,
        'query': {'step': load_step} if load_step is not None else None
    }

    save_to_gfs += [k+'_var' for k in save_to_gfs]
    save_to_gfs += val_targets_list + ['rgb']

    group = 'tr' if group == 'train' else group
    save_id = exp_id if save_id is None else save_id
    params['save_params'] = {
        'exp_id': prefix + save_id + group + '_' + str(load_step),
        'save_to_gfs': save_to_gfs + model_params.get('target_layers', []),
        'port': port
    }

    # data sources
    if data_params.get('sources', None) is None:
        data_params['sources'] = [IMAGES, DEPTHS, NORMALS, OBJECTS] + [t for t in val_targets_list if '_mask' not in t]

    # validation params
    params['validation_params'] = {}
    for dataset in dataset_names:
        train_data_params, val_data_params = get_data_params(dataset_names=[dataset],
                                                             batch_size=batch_size,
                                                             sequence_len=sequence_len,
                                                             dataprefix=dataprefix,
                                                             n_tr_per_dataset=NUM_VAL_EXAMPLES,
                                                             n_val_per_dataset=NUM_VAL_EXAMPLES,
                                                             shuffle_val=shuffle_val,
                                                             shuffle_seed=shuffle_seed,
                                                             **data_params)
        train_data_params['batch_size'] = batch_size
        val_data_params['batch_size'] = batch_size
        def _data_input_fn_wrapper(batch_size, **kwargs):
            data_provider_cls = TdwSequenceDataProvider(**kwargs)
            return data_provider_cls.input_fn(batch_size)
        train_data_params['func'] = _data_input_fn_wrapper
        val_data_params['func'] = _data_input_fn_wrapper

        try:
            val_prefix = prefix + '_' + dataset.split('_')[-2] # better naming convention for new tdw data
        except IndexError:
            try:
                val_prefix = prefix + '_' + dataset.split('/')[-1]
            except IndexError:
                val_prefix = prefix + '_' + dataset
        print("val key", val_prefix)
        train_data_params['is_training'] = False

        params['validation_params'][val_prefix] = {
            'data_params': train_data_params if group == 'tr' else val_data_params,
            'targets': {
                'func': target_func,
                'target': val_targets_list
            },
            'queue_params': None,
            'num_steps': num_steps,
            'agg_func': agg_func,
            'online_agg_func': online_agg_func
        }
        for k,v in target_func_kwargs.items():
            params['validation_params'][val_prefix]['targets'][k] = v

        if pickle_results:
            local_pickle_path = local_pickle_dir + '/' + save_id + '.pkl'
            def _pickle_agg_func(res):
                return eval_metrics.just_keep_everything(pickle_results_func(res, local_pickle_path))
            params['validation_params'][val_prefix]['agg_func'] = _pickle_agg_func

    tfutils.base.test_from_params(**params)

if __name__ == '__main__':

    encoder_params_train = {

        # temporal
        'inp_sequence_len': 5,
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
        'num_levels': None
    }

    # dynamics model and rollout
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

    # data params
    filter_func = (moving_and_any_inview_and_not_acting_and_not_teleporting_func, ['is_moving3', 'is_object_in_view', 'is_acting', 'is_not_teleporting'])

    tdw_data_params = {'train_filter_rule': None, 'val_filter_rule': None,
                       'enqueue_batch_size': 10, 'delta_time':1, 'buffer_mult':10,
                       'motion_filter': False, 'motion_thresh': 0.03, 'motion_area_thresh': 0.1,
                       'resizes': RESIZES, 'get_segments': True
    }

    if train_net:
        train_vectorizing_and_dynamics_model(
            # load/save
            DEFAULT_INTEGRATED_PARAMS,
            port=27024,
            train_on_tdw=True,
            restore_global_step=True,
            load=False,
            load_step=None,
            save_valid_freq=10000,
            save_filters_freq=10000,

            # model params
            sequence_len=5,
            model_params=encoder_params_train,
            dynamics_model=PhysicsModel,
            dynamics_model_params=dynamics_model_params,
            prediction_kwargs=prediction_kwargs,

            # naming experiment
            dbname='dyn',
            collname='gtinp',

            # list of experiments
            exp_id='seq5del1_playV1_1', #a3

            # datasets
            # dataprefix='/mnt/fs4/dbear/datasets/',
            # dataset_names=['ball_hits_primitive3_experiment'],
            dataprefix='/data4/dbear/tdw_datasets/',
            dataset_names=['playroom_v1'],

            # training and validation params
            learning_rate=2e-4,
            loss_func=end_to_end_loss,
            loss_func_kwargs={'dynamics_loss_scale': 1000.0},

            validate_first=False,
            validate_object_metrics=False,
            validate_node_attributes=False,
            validation_func=None,

            # data params
            data_params=tdw_data_params,
            camera_normals=True
        )


    else: # Evaluation
        encoder_params_val = copy.deepcopy(encoder_params_train)
        prediction_kwargs_val = copy.deepcopy(prediction_kwargs)

        validate_integrated_model(
            dbname='dyn', collname='gtinp',
            # 'dyn', 'gtinp', 'seq5del2_1var1spa1bord_1pusht_Lpos0sg0bg_cs3',
            # 'dyn', 'gtinp', 'seq5del2_1var1spa1bord_1colladdt1pusht_Lpos0sg0bg_prim3',
            # 'dyn', 'gtinp', 'seq5del2_1var1spa1bord_1pusht_Lpos0sg0bg_prim3',
            # 'dyn', 'gtinp', 'seq5del2_1var1spa1bord_0pusht_Lpos0sg0bg_prim3',
            # 'dyn', 'gtinp', 'seq5del2_1var1spa1bord_1colladdt1pusht_ro-1_Lpos0sg0bg_prim3',
            # exp_id='seq5del2_1var1spa1bord_1h3colladdlart1pusht_hist3_prim3', load_step=450000, # z3
            exp_id='seq5del2_1var1spa1bord_1h3colladdlart1pusht_hist3_prim3', load_step=750000, # z3
            # exp_id='seq5del2_1var1spa1bord_1colladdlart1pusht_Lpos0sg0bg_prim3', load_step=500000, # t3
            # exp_id='seq5del2_1var1spa1bord_1pusht_hist3_prim3', load_step=750000,
            save_id=None, prefix='loss1', load_port=27023, port=27023, batch_size=10, num_steps=500,
            model_params=encoder_params_val, sequence_len=(encoder_params_val['inp_sequence_len']+encoder_params_val.get('future_rollout_times', 0)), shuffle_seed=0, data_params=tdw_data_params, val_on_tdw=True, camera_normals=True,
            dynamics_model=PhysicsModel, dynamics_model_params=copy.deepcopy(dynamics_model_params),
            prediction_kwargs=prediction_kwargs_val,
            # dataprefix='/mnt/fs4/cfan/tdw-agents/', dataset_names=['cube_sphere_1000'], group='val',
            dataset_names=['ball_hits_primitive2_experiment'], group='val',
            target_func=store_outputs,
            # target_func_kwargs={'target_layers': ['nodes_level_0', 'nodes_level_1', 'segments_level_1', 'dynamics_loss', 'nodes_level_0_pred', 'nodes_level_1_pred']}
            target_func_kwargs={'target_layers': ['dynamics_loss'], 'store_labels':False},
            online_agg_func=eval_metrics.append_each_val, agg_func=eval_metrics.agg_mean_and_var
            # target_func=store_input_images
        )
