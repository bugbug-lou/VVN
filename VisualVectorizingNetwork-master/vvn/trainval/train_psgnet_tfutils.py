from __future__ import division, print_function, absolute_import

import dill
import imp
import os
import pprint
import copy

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

from tfutils import base, optimizer

# from vvn.data.data_utils import get_data_params
from vvn.data.tdw_data import TdwSequenceDataProvider
import vvn.models.psgnet as psgnet
from utils import collect_and_flatten
from training_configs import DEFAULT_TFUTILS_PARAMS

import pdb

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "dbname", "vvn", help="Name of database to store in mongodb")
flags.DEFINE_string(
    "collname", "psgnet", help="Name of collection to store in mongodb")
flags.DEFINE_string(
    "exp_id", None, help="Name of experiment to store in mongodb")
flags.DEFINE_integer(
    "port", 27024, help="localhost port of the mongodb to save to")
flags.DEFINE_string(
    "config_path", None, help="Path to the config for a VVN model")
flags.DEFINE_integer(
    "batch_size", 1, help="batch size for model")
flags.DEFINE_integer(
    "sequence_length", 4, help="movie sequence length for model")
flags.DEFINE_string(
    "data_dir", None, help="the directory with various datasets"),
flags.DEFINE_string(
    "dataset_names", "playroom_v1", help="The comma-separated names of the TDW datasets to train from")
flags.DEFINE_integer(
    "minibatch_size", None, help="minibatch size for model")
flags.DEFINE_integer(
    "num_gpus", 1, help="number of gpus to run model on")
flags.DEFINE_bool(
    "use_default_params", True, help="Use the default tfutils train params other than model, saving, etc.")
flags.DEFINE_string(
    "gpus", None, help="GPUs to train on")
flags.DEFINE_bool(
    "train", True, help="whether to train or test")
flags.DEFINE_bool(
    "load", False, help="whether to load a previous model saved under this name")
flags.DEFINE_string(
    "load_exp_id", None, help="name of exp to load from")
flags.DEFINE_integer(
    "step", None, help="Which step to load from"),
flags.DEFINE_string(
    "trainable", None, help="Comma-separated list of trainable scope names. Default trains all variables")
flags.DEFINE_integer(
    "seed", 0, help="random seed")
flags.DEFINE_string(
    "save_dir", None, help="where to save a pickle of the tfutils_params")

flags.mark_flag_as_required("exp_id")
flags.mark_flag_as_required("config_path")
flags.mark_flag_as_required("gpus")

def load_config(config_path):
    config_path = os.path.abspath(config_path)
    if config_path[-3:] != ".py":
        config_path += ".py"
    config = imp.load_source('config', config_path)
    logging.info("Config loaded from %s" % config_path)
    default_params = copy.deepcopy(DEFAULT_TFUTILS_PARAMS)
    config = config.config
    config['config_path'] = config_path
    config['default_params'] = default_params
    return config

def update_tfutils_params(which_params='save', base_params={}, new_params={}, config={}):

    if new_params is None:
        return
    key = which_params + '_params'
    params = copy.deepcopy(base_params.get(key, {}))
    config_params = copy.deepcopy(config.get(key, {}))
    config_params.update(new_params)
    params.update(config_params)
    base_params[key] = params
    return

def build_trainval_params(config, loss_names=[]):
    data_params = config.get('data_params', {'func': TdwSequenceDataProvider})
    data_provider_cls = data_params['func']
    def _data_input_fn_wrapper(batch_size, train, **kwargs):
        data_provider = data_provider_cls(**kwargs)
        return data_provider.input_fn(batch_size, train)

    train_data_params, val_data_params = data_provider_cls.get_data_params(
        batch_size=FLAGS.batch_size,
        sequence_len=FLAGS.sequence_length,
        dataprefix=FLAGS.data_dir or data_provider_cls.DATA_PATH,
        dataset_names=FLAGS.dataset_names.split(','),
        **data_params)

    train_data_params.update({'func': _data_input_fn_wrapper, 'batch_size': FLAGS.batch_size, 'train': True})
    val_data_params.update({'func': _data_input_fn_wrapper, 'batch_size': FLAGS.batch_size, 'train': False})

    train_params_targets = {
        'func': collect_and_flatten,
        'targets': loss_names
    }
    train_params = {
        'minibatch_size': FLAGS.minibatch_size or FLAGS.batch_size,
        'data_params': train_data_params,
        'targets': train_params_targets,
    }
    val_params = config.get('validation_params', {})
    for val_key, val_dict in val_params.items():
        val_dict.update({
            'data_params': val_data_params,
            'num_steps': val_params[val_key].get('val_length', 50000) // FLAGS.batch_size
        })

    return train_params, val_params

def initialize_psgnet_model(config):
    model_params = config['model_params']
    Model = psgnet.PSGNet(**model_params)
    model_call_params = copy.deepcopy(Model.params)
    model_call_params['func'] = Model
    logging.info(pprint.pformat(model_params))
    logging.info(pprint.pformat(model_call_params))
    return model_call_params

def save_config(tfutils_params, save_dir=None):
    save_params = tfutils_params['save_params']
    fname = save_params['dbname'] + '.' + save_params['collname'] + '.' + save_params['exp_id'] + '.pkl'
    if save_dir is not None:
        with open(os.path.join(save_dir, fname), 'wb') as f:
            dill.dump(tfutils_params, f)
            f.close()

def train(config, dbname, collname, exp_id, port, gpus=[0], use_default=True, load=True):

    tfutils_params = config['default_params'] if use_default else {}

    ### MODEL ###
    model_params = initialize_psgnet_model(config)
    loss_names = model_params['func'].Losses.keys()
    model_params.update({
        'devices': ['/gpu:' + str(i) for i in range(len(gpus))],
        'num_gpus': len(gpus),
        'seed': FLAGS.seed,
        'prefix': 'model_0'
    })
    tfutils_params['model_params'] = model_params

    ### INPUT DATA ###
    train_params, val_params = build_trainval_params(config, loss_names=loss_names)
    update_tfutils_params('train', tfutils_params, train_params, config={})
    update_tfutils_params('validation', tfutils_params, val_params, config={})

    ### OPTIMIZATION ###
    trainable = FLAGS.trainable
    if trainable is not None:
        trainable = trainable.split(',')
    opt_params = {'trainable_scope': trainable}
    update_tfutils_params('optimizer', tfutils_params, opt_params, config)
    update_tfutils_params('loss', tfutils_params, {}, config)
    update_tfutils_params('learning_rate', tfutils_params, {}, config)

    ### SAVE AND LOAD ###
    save_params = {
        'dbname': dbname,
        'collname': collname,
        'exp_id': exp_id,
        'port': port
    }
    update_tfutils_params('save', tfutils_params, save_params, config)

    load_params = copy.deepcopy(save_params)
    load_exp_id = FLAGS.load_exp_id or exp_id
    load_params.update({
        'do_restore': True,
        'exp_id': load_exp_id,
        'query': {'step': FLAGS.step},
        'restore_global_step': True if (exp_id == load_exp_id) else False
    })
    update_tfutils_params('load', tfutils_params, load_params if load else None, config)

    ### TODO save out config ###
    save_config(tfutils_params, save_dir=FLAGS.save_dir)

    logging.info(pprint.pformat(tfutils_params))
    base.train_from_params(**tfutils_params)

def test(config, use_default=True):
    pass

def main(argv):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
    tf.logging.set_verbosity(tf.logging.ERROR)
    config = load_config(FLAGS.config_path)
    gpus = FLAGS.gpus.split(',')
    if FLAGS.train:
        train(config, FLAGS.dbname, FLAGS.collname, FLAGS.exp_id, FLAGS.port,
              gpus=gpus, use_default=FLAGS.use_default_params, load=FLAGS.load)
    else:
        test(config, use_default=FLAGS.use_default_params)

if __name__ == '__main__':
    app.run(main)
