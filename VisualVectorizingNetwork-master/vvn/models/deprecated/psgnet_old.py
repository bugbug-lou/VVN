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
from vvn.models import rendering
from vvn.models.decoding import GraphDecoder
from vvn.models.extractor import Extractor, CNN, ConvRNN

## for debugging
from vvn.data.tdw_data import TdwSequenceDataProvider
from vvn.data.data_utils import *


class GraphLevel(object):
    '''
    An object for storing a graph's nodes, edges, and attributes
    '''
    def __init__(self):
        pass

class PSGConfig(object):
    '''
    An object for designing the PSG structures produced by a PSGNet
    '''
    def __init__(self):
        pass

class PSG(Graph):
    '''
    A graph with hierarchical structure, spatial registrations, and methods
    for postprocessing, deriving, and comparing named physical attributes (e.g. color, texture)
    '''
    def __init__(self):
        pass

class PSGNet(object):
    '''
    A model class that takes a dict of image-like tensors as input and learns a hierarhical
    graph representation on top of convolutional features extracted from them.
    '''
    def __init__(
            self,
            model_name=None,
            dimensions={},
            preprocessing_func=utils.concat_and_name_tensors,
            preprocessing_kwargs={'tensor_order': ['images']},
            feature_extractor=(Extractor(), {}, {}),
            graph_config=PSGConfig(),
            decoders=[(GraphDecoder, {'name':'decoder'}, {})],
            **kwargs
    ):

        self.model_name = model_name or "PSGNet"
        self.DimsIn = DimensionDict(**dimensions)
        self.DimsOut = DimensionDict()

        ## for converting input dict into image-like tensor [B,T,H,W,C]
        ## signature image_tensor = func(input_dict, dimensions=None, **preprocessing_kwargs)
        self.preproc = preprocessing_func
        self.preproc_kwargs = copy.deepcopy(preprocessing_kwargs)

        ## extract features [B,T',H',W',C']
        ## signature feature_tensor = extractor(image_tensor, is_training, **extraction_kwargs)
        self.extractor = feature_extractor[0](**feature_extractor[1])
        self.extractor_kwargs = copy.deepcopy(feature_extractor[2])

        ## config for going from features to a hierarchical graph structure
        self.graph_config = graph_config

        ## initialize decoders and prediction kwargs
        self.decoders = [D[0](**D[1]) for D in decoders]
        self.decoder_names = [D.name for D in self.decoders]
        assert len(self.decoder_names) == len(set(self.decoder_names)), "Decoders must have unique names"
        self.prediction_kwargs = {D.name: decoders[i][2] for i,D in enumerate(self.decoders)}


    def preprocess_inputs(self, input_dict, **kwargs):
        '''
        Apply the model's preprocessing func and update the tensor dimension dict

        input_dict: a <dict> of tensors that have the same first four dimensions [B,T,H,W]

        returns:
        model_inputs: [B,T,H,W,C] <tf.float32> tensor

        modifies in place:
        self.Dims
        '''

        with tf.compat.v1.variable_scope(self.model_name + '/Preprocessing'):
            model_inputs = self.preproc(input_dict, dimensions=self.DimsIn, **self.preproc_kwargs)
        return model_inputs

    def extract_features(self, inputs, train=True, **kwargs):

        with tf.compat.v1.variable_scope(self.model_name + '/Extraction'):
            inputs = self.extractor(inputs, train, **self.extractor_kwargs)
            self.Hf,self.Wf,self.Cf = inputs.shape.as_list()[-3:]
            self.DimsOut['features'] = [0,inputs.shape.as_list()[-1],None]
        return inputs

    def build_psg(self, base_tensor, graph_config, **kwargs):
        return Graph()

    def decode(self, PSG_or_features, train=True, graph=True, **kwargs):

        preds = {}
        with tf.compat.v1.variable_scope(self.model_name + '/Decoding'):
            for Decoder in self.decoders:
                if graph == ('Graph' in type(Decoder).__name__):
                    print("graph", graph, "decoder", Decoder, Decoder.name, Decoder.params)
                    preds.update(Decoder(PSG_or_features, train, **self.prediction_kwargs[Decoder.name]))
        return preds

    def update_input_shapes(self, inputs, inp_sequence_len=None):
        if inp_sequence_len is not None:
            inputs = inputs[:,:inp_sequence_len]
        B,T,H,W,C = inputs.shape.as_list()
        BT,HW,_,R = utils.dims_and_rank(inputs)
        self.B,self.T,self.H,self.W,self.C = B,T,H,W,C
        self.BT,self.HW,self.R = BT,HW,R
        return inputs

    def __call__(
            self,
            inputs,
            train=True,
            train_targets=[],
            inp_sequence_len=None,
            **model_kwargs
    ):

        # get labels
        labels = {k: inputs[k] for k in train_targets}

        # preprocess the inputs
        inputs = self.preprocess_inputs(inputs)
        inputs = self.update_input_shapes(inputs, inp_sequence_len)

        # extract features
        base_tensor = self.extract_features(inputs, train)
        features = self.extractor.outputs # dict

        print("base_tensor")
        print(base_tensor)

        print("extracted outputs")
        print(features)

        # build graph
        PSG = self.build_psg(base_tensor, self.graph_config, **model_kwargs)
        print("PSG", PSG)

        # decode features and graph states into predictions
        feature_preds = self.decode(features, train=train, graph=False)
        graph_preds = self.decode(PSG, train=train, graph=True)

        print("feature preds", feature_preds)
        print("graph preds", graph_preds)

        # get losses


if __name__ == '__main__':

    SHAPE = [2,4,64,64]
    B,T,H,W = SHAPE

    inputs = {
        'images': tf.random.normal(SHAPE + [3], dtype=tf.float32),
        'depths': tf.random.normal(SHAPE + [1], dtype=tf.float32),
        'normals': tf.random.normal(SHAPE + [3], dtype=tf.float32),
        'objects': tf.random_uniform(SHAPE + [3], minval=0, maxval=8, dtype=tf.int32)
    }

    # add diff images
    inputs['diff_images'] = utils.image_time_derivative(inputs['images'])
    inputs['hw'] = utils.coordinate_ims(B,T,[H,W])

    model = PSGNet(model_name='PSGNetM',
                   preprocessing_func=utils.preproc_tensors_by_name,
                   preprocessing_kwargs={'dimension_order': ['images', 'normals', 'diff_images', 'hw'],
                                         'dimension_names':{'images':'rgb'},
                                         'preproc_funcs': {
                                             'rgb': lambda rgb: tf.cast(rgb, tf.float32) / 255.,
                                             'normals': lambda n: tf.nn.l2_normalize(tf.cast(n, tf.float32) / 255., axis=-1)
                                         }
                   },
                   feature_extractor=(Extractor, {
                       'model_func':convolutional.convnet_stem,
                       'ksize':7, 'conv_kwargs':{'activation': 'relu'}, 'max_pool':True}, {})
    )

    model(inputs, train=True, train_targets=['objects'])
    print("input Dims", model.DimsIn)
    print("output Dims", model.DimsOut)
