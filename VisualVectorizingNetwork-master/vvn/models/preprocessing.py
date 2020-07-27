import tensorflow as tf

from vvn.ops.dimensions import DimensionDict, OrderedDict
from .base import Model

def preproc_rgb(img, norm=255.):
    if img.dtype == tf.uint8:
        return tf.cast(img, tf.float32) / norm
    else:
        return img

def concat_and_name_tensors(tensor_dict, tensor_order=[], dimensions=None, dimension_names={},
                            **kwargs):

    assert isinstance(tensor_dict, dict)
    assert all([isinstance(v, tf.Tensor) for v in tensor_dict.values()]), "Must pass a dict of all tensors"
    if dimensions is None:
        dimensions = DimensionDict()

    out = []
    for nm in tensor_order:
        tens = tensor_dict[nm]
        out.append(tens)
        dimensions[dimension_names.get(nm, nm)] = tens.shape.as_list()[-1]

    return tf.concat(out, axis=-1)

def preproc_tensors_by_name(
        tensor_dict,
        dimension_order=['images'],
        dimension_preprocs={'images': lambda rgb: tf.cast(rgb, tf.float32) / 255.},
        dimensions=None, dimension_names={}, **kwargs
):
    '''
    For each tensor in tensor_dict named by a key in tensor_order, apply associated preproc func.
    Then concat the results and update the DDict passed by dimensions

    '''
    tensor_names = [dimension_names.get(nm, nm) for nm in dimension_order]
    funcs = {nm: dimension_preprocs.get(nm, lambda data: tf.identity(tf.cast(data, tf.float32), name=nm+'_preproc'))
             for nm in tensor_names}

    ## create the tensor and dimensions
    tensor = concat_and_name_tensors(tensor_dict, dimension_order, dimensions, dimension_names, **kwargs)

    ## update dim preprocs
    dimensions.set_postprocs(funcs)

    ## apply the postprocs
    preproc_tensor = dimensions.get_tensor_from_attrs(
        tensor, tensor_names, postproc=True, stop_gradient=kwargs.get('stop_gradient', False))

    return preproc_tensor

class Preprocessor(Model):

    def __init__(
            self,
            model_func=preproc_tensors_by_name,
            **kwargs
    ):

        self.dims = None
        super(Preprocessor, self).__init__(model_func, **kwargs)

    def __call__(self, inputs, train=True, **kwargs):
        assert isinstance(inputs, dict), "Must pass a dict of tensors to the preprocessor"
        assert all((isinstance(val, tf.Tensor) for val in inputs.values())), "Must pass a dict of tensors to preprocssor"

        self.dims = DimensionDict()
        outputs = super(Preprocessor, self).__call__(
            inputs, dimensions=self.dims, **kwargs)

        assert self.dims.ndims == outputs.shape.as_list()[-1],\
            "All dims must be logged in Preprocessor.dims but outputs shape = %s and dims = %s"\
            % (outputs.shape.as_list(), self.dims)

        return outputs
