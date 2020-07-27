from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from collections import OrderedDict

class DimensionDict(OrderedDict):
    """
    A dictionary that keeps track of which vector dimension (channels) correspond to which attributes

    Also allows postproc functions to be attached to each attribute
    """
    # __getattr__ = dict.get # can get vals with dot notation

    def __init__(self, *args, **kwargs):

        super(DimensionDict, self).__init__()

        if len(args) == 0:
            self.ndims = 0
        elif len(args) == 1:
            if isinstance(args[0], int):
                self.ndims = args[0]
            else:
                self.ndims = 0
                assert all([(v[0] if isinstance(v, (list,tuple)) else v) >= 0 for k,v in args[0].items()]),\
                "Can't initialize an empty DimensionDict with negative indexing"
                self.update(*args, **kwargs)
        elif len(args) == 2:
            assert isinstance(args[0], int)
            self.ndims = args[0]
            assert all([(v[0] if isinstance(v, (list,tuple)) else v) >= -self.ndims for k,v in args[1].items()]),\
            "Can't initialize an empty DimensionDict with negative indexing"
            self.update(*args[1:], **kwargs)
        else:
            raise ArgumentError("DimensionDict initialized with at most an int and an iterable")



    def __setitem__(self, key, value):

        # handle cases
        # if it's an int, append value dimensions at the end
        if isinstance(value, str):
            assert value in self.keys(), "Can't assign %s to %s because %s isn't in dict!" % (key, value, value)
            super(DimensionDict, self).__setitem__(key, self[value])

        elif isinstance(value, type(tf.identity)) or value is None:
            assert key in self.keys()
            dims = self[key]
            super(DimensionDict, self).__setitem__(key, [dims[0], dims[1], value])

        elif isinstance(value, int):
            assert value >= 0, "Number of dimensions to add must be nonnegative"
            value = [self.ndims, self.ndims + value, None]
            self.ndims = value[1]
            super(DimensionDict, self).__setitem__(key, value)

        elif isinstance(value, (list, tuple)):
            assert len(value) in [2,3], "DimensionDict values must be (dstart, dend, [postproc])"
            assert all([isinstance(d, int) for d in value[:2]]), "first two values must be dstart, dend"
            ds, de = value[:2]
            dnew = de - ds
            assert (dnew >= 0)
            assert value[0]*value[1] >= 0, "start and end dims must both be nonnegative or nonpositive"

            if ds + dnew > self.ndims:
                self.ndims = self._dim(ds) + dnew # add new dims to the total
            value = self._dim([ds, de]) + value[2:]
            if len(value) == 2:
                value += [None]

            super(DimensionDict, self).__setitem__(key, value)

    def _dim(self, d):
        if isinstance(d, str):
            return d
        elif isinstance(d, type(tf.function)) or d is None:
            return d
        elif isinstance(d, (list, tuple)):
            return [0 if d[0] == 0 else self._dim(d[0]), self._dim(d[1])] + d[2:]
        else:
            assert isinstance(d, int)
            assert d >= -self.ndims, "Can't negative index more than ndims"

            if d < 0:
                return d % self.ndims
            elif d == 0:
                return self.ndims
            else:
                return d

    def get_unassigned_dims(self):
        used = [False]*self.ndims
        for attr, dims in self.items():
            for i in range(dims[0],dims[1]):
                used[i] = True
        used += [True]

        unassigned = []
        s,e = 0,0
        for c in range(self.ndims+1):
            if used[c]:
                if s < e:
                    unassigned.append([s,e])
                s = c+1
                e = c+1
            else:
                e = c+1

        return unassigned

    def delete(self, name, remove_dims=False):
        if remove_dims:
            print("removing dims from", name)
            raise NotImplementedError("Should remove all dims that overlap with %s" % name)
        super(DimensionDict, self).__delitem__(name)

    def insert(self, name, *args, **kwargs):
        func = kwargs.get('func', None)
        if len(args) == 1:
            if isinstance(args[0], int):
                self[name] = args[0]
                return
            assert isinstance(args[0], (list, tuple))
            assert len(args[0]) in [2,3]
            ds, de = self._dim(args[0][:2])
            dnew = de - ds
            if len(args[0]) == 3:
                assert func is None
                func = args[0][2]
        elif len(args) in [2,3]:
            ds, dnew = args[:2]
            ds,_ = self._dim([ds,0])
            de = ds + dnew
            if len(args) == 3:
                assert func is None
                func = args[2]
        else:
            raise ArgumentError()

        assert dnew > 0
        if self.ndims == 0:
            self[name] = dnew
            return

        # find which attrs must be bumped up
        self.sort()
        to_bump = [nm for nm,dims in self.items() if ds < dims[1]]
        min_dim = min([self[nm][0] for nm in to_bump]) if len(to_bump) else self.ndims
        to_bump = [nm for nm,dims in self.items() if dims[0] >= min_dim]
        dnew = max([de - min_dim, dnew])

        self.ndims += dnew
        for nm in to_bump:
            self[nm][0] += dnew
            self[nm][1] += dnew

        self[name] = [ds, de, func]

    def insert_from(self, dims, position=None):
        if isinstance(dims, type(self)):
            pos = self.ndims if position is None else position
            pos,_ = self._dim([pos, 0])
            self._merge_ddict(dims, pos)
        elif isinstance(dims, (list, OrderedDict)):
            self._insert_from_list(dims)
        elif isinstance(dims, dict):
            self._insert_from_dict(dims)
        else:
            raise TypeError

    def _merge_ddict(self, ddict, position):
        assert isinstance(position, int)
        self.insert("alloc", [position, position + ddict.ndims])
        _ = self.pop("alloc")
        for k,v in ddict.items():
            self[k] = [v[0] + position, v[1] + position] + v[2:]

    def _insert_from_list(self, dims):
        assert isinstance(dims, (tuple, list)), "tried to insert a list of dims but argument was a %s" % type(dims)
        for kv in dims:
            assert isinstance(kv, (tuple, list))
            assert isinstance(kv[0], str)
            assert isinstance(kv[1], (int, list, tuple))
            self.insert(kv[0], kv[1])

    def _insert_from_dict(self, dims):
        assert isinstance(dims, dict)
        dims = reversed(sorted(dims.items(), key=lambda t:t[1]))
        for kv in dims:
            self.insert(kv[0], kv[1])

    def sort(self):
        kwargs = {}
        for (k,v) in list(self.items()):
            kwargs[k] = self.pop(k)
        self.update(kwargs)
        return self

    def update(self, *args, **kwargs):
        if args:
            assert not len(kwargs.keys()), "Update with either args or kwargs, not both"
            if len(args) > 1:
                raise TypeError('update expected at most 1 arguments, got %d' % len(args))

            if isinstance(args[0], OrderedDict):
                kvs = args[0].items()
            else:
                kvs = sorted(args[0].items(), key=lambda t: t[1])
            for kv in kvs:
                self[kv[0]] = self._dim(kv[1])
        else:
            kwargs = sorted(kwargs.items(), key=lambda t: t[1])
            for kv in kwargs:
                self[kv[0]] = self._dim(kv[1])

    def copy(self, suffix='_copy'):
        self.sort()
        keys = []
        values = []
        for kv in self.items():
            keys.append(kv[0] + suffix)
            values.append([v for v in kv[1]])
        ddcopy = DimensionDict({keys[i]:values[i] for i in range(len(keys))})
        ddcopy.sort()
        return ddcopy

    def set_postprocs(self, attr_postprocs):
        assert isinstance(attr_postprocs, dict)
        for attr,func in attr_postprocs.items():
            assert isinstance(func, type(tf.identity)) or (func is None)
            self[attr] = func

    def get_postprocs(self):
        return {attr: self[attr][2] for attr in self.keys()}

    def get_latent_vector(self, tensor):
        dims_list = self.get_unassigned_dims()
        return tf.concat([tensor[...,d[0]:d[1]] for d in dims_list], axis=-1)

    def get_tensor_from_attrs(self, tensor, attr_list, postproc=False, stop_gradient=False, concat=True):
        assert isinstance(tensor, tf.Tensor)
        if isinstance(attr_list, str):
            attr_list = [attr_list]

        if not isinstance(postproc, dict):
            postproc = {attr:postproc for attr in attr_list}
        if not isinstance(stop_gradient, dict):
            stop_gradient = {attr:stop_gradient for attr in attr_list}

        tensor_list = []
        for attr in attr_list:
            dims = self[attr][:2]
            _func = tf.identity
            if postproc.get(attr, False):
                _func = self[attr][2] or (lambda t: tf.identity(t, name='%s_id_postproc'%attr))
            func = (lambda t: tf.stop_gradient(_func(t))) if stop_gradient.get(attr, False) else _func
            tns = func(tensor[...,dims[0]:dims[1]])
            # print("vectorizing %s" % attr, tns)
            tensor_list.append((attr,tns))

        if concat:
            return tf.concat([t[1] for t in tensor_list], axis=-1)
        else:
            return OrderedDict(tensor_list)

    def append_attr_to_vector(self, name, attr_tensor, base_tensor=None):
        if base_tensor is None:
            base_tensor = tf.zeros_like(attr_tensor[...,0:0])
        self.insert(name, self.ndims, attr_tensor.shape.as_list()[-1])
        return tf.concat([base_tensor, attr_tensor], axis=-1)

    def extend_vector(self, tensor_list, base_tensor=None):
        '''
        Add dims specificed by tensor_list
        tensor_list: list of (attr_name <str>, tensor <tf.Tensor>) tuples that match in shape[:-1] with base_tensor
        '''
        if base_tensor is None:
            base_tensor = tf.zeros_like(tensor_list[0][1][...,0:0])
        self.insert_from([(t[0], t[1].shape.as_list()[-1]) for t in tensor_list])
        return tf.concat([base_tensor] + [t[1] for t in tensor_list], axis=-1)

if __name__ == "__main__":

    D = DimensionDict()
    D['rgb'] = [0,3]
    D['depth'] = 1
    D['normals'] = 3
    D.insert('position', 0, 3)
    print(D)
