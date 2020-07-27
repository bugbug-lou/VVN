from __future__ import division, print_function, absolute_import
from collections import OrderedDict
import os
import sys
import pdb

import numpy as np
import tensorflow as tf
import copy

from vvn.ops.dimensions import DimensionDict
from vvn.ops import convolutional
from .base import Model

# for ConvRNNs
import tnn

class Extractor(Model):

    def __init__(self, name=None, base_tensor_name='outputs', layer_names=['outputs'], time_shared=True, **model_params):
        self.name = name or type(self).__name__
        self.base_nm = base_tensor_name
        self.layer_names = layer_names
        self.time_shared = time_shared
        super(Extractor, self).__init__(name=self.name, **model_params)
    def __call__(self, images, train=True, **kwargs):

        assert isinstance(images, tf.Tensor), "Extraction must begin from a tensor"
        assert len(images.shape.as_list()) == 5, "Input tensor must have rank 5 with shape [B,T,H,W,C]"
        if self.time_shared:
            B,T = images.shape.as_list()[0:2]
            shape = images.shape.as_list()[2:]
            images = tf.reshape(images, [B*T] + shape)

        outputs = super(Extractor, self).__call__(
            images, train=train, **kwargs)
        if isinstance(outputs, tf.Tensor):
            outputs = {'outputs': outputs}
            base_tensor = outputs['outputs']
        elif isinstance(outputs, (list, tuple)):
            assert len(outputs) == 2, "If Extractor returns multiple outputs, must be (tf.Tensor, dict)"
            assert isinstance(outputs[0], tf.Tensor) and isinstance(outputs[1], dict), "Extractor model must return tensor and dict"
            output_tensor = outputs[0]
            output_dict = outputs[1]
            output_dict['outputs'] = output_dict.get(self.base_nm, output_tensor)
            outputs = output_dict
            base_tensor = outputs['outputs']
        elif isinstance(outputs, dict):
            try:
                base_tensor = outputs[self.base_nm]
            except KeyError:
                base_tensor = outputs[outputs.keys()[0]]
            outputs['outputs'] = base_tensor

        # only take chosen layers as features
        self.outputs = {nm: outputs.get(nm, None) for nm in self.layer_names + ['outputs']}

        assert isinstance(base_tensor, tf.Tensor), "Base tensor must be a tensor"
        assert all((isinstance(val, tf.Tensor) for val in self.outputs.values())), "All outputs must be tensors %s" % self.outputs

        if self.time_shared:
            base_tensor = tf.reshape(
                base_tensor, [B,T] + base_tensor.shape.as_list()[1:])
            self.outputs = {
                k: tf.reshape(out, [B,T] + out.shape.as_list()[1:])
                for k,out in self.outputs.items()}

        return base_tensor

class CNN(Extractor):

    def predict(self, inputs, train, output_layer=None, **kwargs):
        layer_dict = {}
        raise NotImplementedError("Implement a vanilla CNN")

class ConvRNN(Extractor):

    def __init__(self,
                 base_json='./convrnn_configs/dummy.json',
                 **model_kwargs
    ):

        model_func = None
        super(ConvRNN, self).__init__(
            model_func=model_func,
            **model_kwargs
        )

    def preprocess_time_series(
            self, image_tensor, time_dilation, num_temporal_splits, **kwargs
    ):
        pass

    def predict(
            self,
            image_tensor,
            **kwargs
    ):
        pass
