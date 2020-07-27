from __future__ import division, print_function, absolute_import
from collections import OrderedDict
import os
import sys
import pdb

import numpy as np
import tensorflow as tf
import copy

# Generic graphs
from graph.common import Graph, propdict, graph_op, node_op, edge_op

# Visual Extraction and Graph Building
from vvn.ops.dimensions import DimensionDict
from vvn.ops import convolutional
from vvn.ops import pooling
from vvn.ops import vectorizing
from vvn.ops import graphical
from vvn.ops import utils
from .base import Model, Loss
from .preprocessing import *
from .extracting import *
from .decoding import *
from .losses import *

## for debugging
from vvn.data.tdw_data import TdwSequenceDataProvider
from vvn.data.data_utils import *

class PSGNet(Model):

    def __init__(
            self,
            preprocessor=(Preprocessor, {}),
            extractor=(Extractor, {}),
            decoders=[(Decoder, {})],
            losses=[(Loss, {})],
            **model_params
    ):
        self.num_models = 0
        self.Preproc = self.init_model(Preprocessor, preprocessor)
        self.Extract = self.init_model(Extractor, extractor)

        # set decoders
        self.Decode = {}
        self.num_decoders = self.num_models
        for decoder in decoders:
            DecoderNew = self.init_model(Decoder, decoder)
            self.Decode[DecoderNew.name] = DecoderNew
        self.decoders = sorted(self.Decode.keys())
        self.num_decoders = self.num_models - self.num_decoders
        assert self.num_decoders == len(set(self.decoders)), "Decoders must have unique names"

        # set losses
        self.Losses = {}
        self.num_losses = self.num_models
        for loss in losses:
            LossNew = self.init_model(Loss, loss)
            self.Losses[LossNew.name] = LossNew
        self.loss_names = sorted(self.Losses.keys())
        self.num_losses = self.num_models - self.num_losses
        assert self.num_losses == len(set(self.loss_names))

        # init values
        self.features = {}
        self.psg = {'nodes': {}, 'edges': {}, 'spatial': {}, 'dims': {}}

        super(PSGNet, self).__init__(**model_params)

    def init_model(self, model_class, params):
        if isinstance(params, dict):
            name = params.get('name', type(model_class).__name__ + '_' + str(self.num_models))
            params['name'] = name
            self.num_models += 1
            return model_class(**params)
        elif isinstance(params, (tuple, list)):
            assert len(params) == 2
            name = params[1].get('name', type(params[0]).__name__ + '_' + str(self.num_models))
            params[1]['name'] = name
            self.num_models += 1
            return params[0](**params[1])

    def update_shapes(self, tensor):
        B,T,H,W,C = tensor.shape.as_list()
        BT,HW,_,R = utils.dims_and_rank(tensor)
        self.B,self.T,self.H,self.W,self.C = B,T,H,W,C
        self.BT,self.HW,self.R = BT,HW,R

    def preprocess_inputs(self, inputs, train_targets, inp_sequence_len, scope='Input', **kwargs):

        with tf.compat.v1.variable_scope(scope):
            self.inputs = inputs
            self.labels = {k: inputs[k] for k in train_targets}
            self.input_tensor = self.Preproc(inputs, self.is_training, **kwargs)
            self.inputDims = self.Preproc.dims
            self.input_sequence_len = inp_sequence_len or self.input_tensor.shape.as_list()[1]
            self.input_tensor = self.input_tensor[:,:self.input_sequence_len]
            self.update_shapes(self.input_tensor)
        return self.input_tensor

    def extract_features(self, input_tensor, scope='Extract', **kwargs):

        with tf.compat.v1.variable_scope(scope):
            self.base_tensor = self.Extract(input_tensor, self.is_training, **kwargs)
            self.features = self.Extract.outputs
            self.Tb, self.Hb, self.Wb, self.Cb = self.base_tensor.shape.as_list()[-4:]
            self.baseDims = DimensionDict({'features':self.Cb})
        return self.base_tensor

    def build_psg(self, base_tensor, scope='GraphBuild', **kwargs):

        ### TODO ###
        with tf.compat.v1.variable_scope(scope):
            nodes_level_0 = tf.reshape(base_tensor, [-1,self.Tb,self.Hb*self.Wb,self.Cb])
            self.psg = Graph(edges={}, nodes={'level0': propdict(
                self.baseDims.get_tensor_from_attrs(nodes_level_0, self.baseDims.keys(), concat=False))})
            self.psg.dims = {'level0': self.baseDims.copy(suffix='')}
            self.psg.spatial = {'level0': utils.inds_image(self.B, self.Tb, [self.Hb, self.Wb])}
        return self.psg

    def flatten_features_nodes_edges(self, rename=True):
        all_outputs = {}
        def _rename(prefix, key):
            return prefix + '/' + str(key) if rename else key
        all_outputs.update({_rename('features', layer): self.features[layer]
                            for layer in self.features.keys()})
        all_outputs.update({_rename('nodes', level): self.psg.nodes[level]
                            for level in self.psg.nodes.keys()})
        all_outputs.update({_rename('edges', edge_set): self.psg.edges[edge_set]
                            for edge_set in self.psg.edges.keys()})

        self.decoder_inputs = all_outputs

    def decode_from(self, name, scope='Decode', **kwargs):

        with tf.compat.v1.variable_scope(scope):
            decode_params = copy.deepcopy(kwargs)
            decode_params.update(self.Decode[name].params)
            decoded_outputs = self.Decode[name](
                self.decoder_inputs, train=self.is_training, **decode_params)
        return decoded_outputs

    def compute_loss(self, name, logits_mapping=None, labels_mapping=None, **kwargs):
        Loss = self.Losses[name]
        loss_params = copy.deepcopy(kwargs)
        loss_params.update(Loss.params)
        if logits_mapping is not None:
            loss_params['logits_mapping'] = logits_mapping
        if labels_mapping is not None:
            loss_params['labels_mapping'] = labels_mapping

        to_decode = Loss.required_decoders + [logits_nm.split('/')[0] for logits_nm in loss_params['logits_mapping'].values()]
        # compute required outputs
        for decoder in to_decode:
            if decoder not in [out_nm.split('/')[0] for out_nm in self.outputs.keys()]:
                decode_kwargs = copy.deepcopy(kwargs)
                decode_kwargs['name'] = decoder
                self.outputs.update(self.decode_from(**decode_kwargs))

        # get required labels
        for label in loss_params['labels_mapping'].values():
            if label not in self.labels.keys():
                try:
                    self.labels[label] = self.decoder_inputs[label]
                except KeyError:
                    try:
                        self.labels[label] = self.outputs[label]
                    except KeyError:
                        self.labels[label] = self.inputs[label]

        # TODO get valid

        # compute the loss

        loss_here = Loss(self.outputs, self.labels, **loss_params)

        return loss_here

    def build_model(self, **model_params):

        def model(inputs, train=True, train_targets=[], inp_sequence_len=None, to_decode=None, losses_now=None, rename=True, **kwargs):

            self.is_training = train
            input_tensor = self.preprocess_inputs(inputs, train_targets, inp_sequence_len, **kwargs)
            base_tensor = self.extract_features(input_tensor, **kwargs)
            psg = self.build_psg(base_tensor, **kwargs)

            # compute outputs necessay for losses
            self.outputs = {}
            self.flatten_features_nodes_edges(rename=rename)
            if train:
                losses = {}
                if losses_now is None:
                    losses_now = [{'name': loss_name} for loss_name in self.loss_names]
                for loss_params in losses_now:
                    losses.update(self.compute_loss(**loss_params))
                self.losses = losses
                return losses, self.params
            else:
                outputs = {}
                if to_decode is None:
                    to_decode = [{'name': decoder_name} for decoder_name in self.decoders]
                for decoder_params in to_decode:
                    outputs.update(self.decode_from(**decoder_params))

                self.outputs = outputs
                return outputs, self.params

        self.model_func = model

if __name__ == '__main__':

    SHAPE = [2,4,64,64]
    B,T,H,W = SHAPE
    TRAIN = True

    from resnets.resnet_model import resnet_v2
    resnet_18 = resnet_v2(18, get_features=True)

    inputs = {
        'images': tf.random.normal(SHAPE + [3], dtype=tf.float32),
        'depths': tf.random.normal(SHAPE + [1], dtype=tf.float32),
        'normals': tf.random.normal(SHAPE + [3], dtype=tf.float32),
        'objects': tf.random.uniform(SHAPE + [3], minval=0, maxval=8, dtype=tf.int32),
        'labels': tf.random.uniform([B,T], minval=0, maxval=1001, dtype=tf.int32)
    }

    # add diff images
    inputs['diff_images'] = utils.image_time_derivative(inputs['images'])
    inputs['hw'] = utils.coordinate_ims(B,T,[H,W])

    M = PSGNet(
        preprocessor=(Preprocessor, {
            'model_func': preproc_tensors_by_name,
            'dimension_order': ['images', 'depths', 'normals', 'diff_images', 'hw'],
            'dimension_preprocs': {'images': preproc_rgb}
        }),
        extractor=(Extractor, {
            'model_func': resnet_18, 'name': 'ResNet18', 'layer_names': ['block'+str(i) for i in range(5)]+['pool'], 'base_tensor_name': 'pool',
            # 'model_func': convolutional.convnet_stem, 'name': 'ConvNet', 'layer_names': ['conv'+str(i) for i in range(5)],
            # 'ksize': 7, 'conv_kwargs': {'activation': 'relu'}, 'max_pool': True,
            # 'hidden_ksizes': [3,3,3,3], 'hidden_channels': [64,128,256,512], 'out_channels': 1024
        }),
        decoders=[
            (Decoder, {'name': 'avg_pool', 'model_func': convolutional.global_pool, 'kind': 'avg', 'keep_dims': True, 'input_mapping':{'inputs':'pool'}}),
            (Decoder, {'name': 'classifier', 'model_func': convolutional.fc, 'out_depth': 1000, 'input_mapping': {'inputs':'pool'}})
        ],
        losses=[
            (Loss, {'name': 'classification', 'required_decoders': ['classifier'], 'scale':10.0, 'loss_func':tf.nn.sparse_softmax_cross_entropy_with_logits, 'logits_keys': ['logits'], 'labels_keys': ['labels']}),
            (Loss, {'name': 'L2', 'loss_func': l2_loss, 'scale':1.0})
        ]
    )

    losses = [
        {'name': 'classification', 'logits_mapping': {'logits': 'classifier/outputs'}, 'labels_mapping': {'labels': 'labels'}},
        {'name': 'L2', 'logits_mapping': {'logits': 'avg_pool/outputs'}, 'labels_mapping': {'labels': 'avg_pool/outputs'}}
    ]

    print("outputs", M(inputs, train=TRAIN, train_targets=['labels'], losses_now=losses, rename=False))
    print("psg", M.psg)
    print("features", M.features)
    import pdb
    pdb.set_trace()
