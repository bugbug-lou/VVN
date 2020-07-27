import numpy as np
import tensorflow as tf
import os
import copy

def read_depths_image(im, mask=None, new=True, normalization=100.1, background_depth=30.0):

    im = tf.cast(im, dtype=tf.float32)
    assert im.shape.as_list()[-1] == 3
    if not new:
        channels = tf.constant([256.0, 1.0, 1.0/256.0], dtype=tf.float32)
        im = tf.reduce_sum(tf.multiply(im, tf.reshape(channels, [1,1,1,1,3])), axis=-1, keepdims=True) / normalization
    elif new:
        channels = tf.constant([256.0*256.0, 256.0, 1.0], dtype=tf.float32)
        im = tf.reduce_sum(tf.multiply(im, tf.reshape(channels, [1,1,1,1,3])), axis=-1, keepdims=True) * \
             tf.divide(normalization, tf.pow(256.0, 3))

    if mask is not None:
        im = mask_tensor(im,
                         mask=tf.cast(mask, dtype=tf.float32),
                         mask_value=background_depth)
    else:
        im = tf.where(im < background_depth,
                      im,
                      background_depth*tf.ones_like(im)) # true
    return im

def read_background_segmentation_mask_new(obj_mask):
    ''' obj_mask = inputs['objects'] from data provider '''
    assert obj_mask.dtype == tf.uint8, "objects mask must be read as tf.uint8"
    assert len(obj_mask.shape.as_list()) == 5, "must pass a [B,T,H,W,3] tensor"
    mask = tf.reduce_sum(obj_mask, axis=4, keepdims=True) # sum along channels. background sum is 0
    binary_mask = tf.cast(mask > 0, dtype=tf.bool)
    return binary_mask

def object_id_hash(objects, dtype_out=tf.int32, val=256):
    '''
    objects: [...,C]
    val: a number castable to dtype_out

    returns:
    out: [...,1] where each value is given by sum([val**(C-1-c) * objects[...,c:c+1] for c in range(C)])
    '''
    if not isinstance(objects, tf.Tensor):
        objects = tf.convert_to_tensor(objects)
    C = objects.shape.as_list()[-1]
    val = tf.constant(val, dtype=dtype_out)
    objects = tf.cast(objects, dtype_out)
    out = tf.zeros_like(objects[...,0:1])
    for c in range(C):
        scale = tf.pow(val, C-1-c)
        out += scale * objects[...,c:c+1]

    return out

def any_inview_func(data, keys):
    assert keys[0] == 'is_object_in_view'
    any_obj_in_view = tf.reduce_max(tf.cast(data[keys[0]], tf.int32), axis=2, keepdims=True)
    any_obj_in_view = tf.cast(any_obj_in_view, tf.bool)
    return any_obj_in_view

def moving_and_any_inview_func(data, keys):
    assert (keys[0] in ['is_moving', 'is_moving2', 'is_moving3']) and (keys[1] == 'is_object_in_view'), keys
    any_obj_in_view = tf.reduce_max(tf.cast(data[keys[1]], tf.int32), axis=2, keepdims=True)
    any_obj_in_view = tf.cast(any_obj_in_view, tf.bool)
    return tf.logical_and(data[keys[0]], any_obj_in_view)

def single_object_inview_func(data, keys):
    objects_in_view = tf.cast(data['is_object_in_view'], tf.int32)
    num_objects_in_view = tf.reduce_sum(objects_in_view, axis=1, keepdims=True)
    return tf.equal(num_objects_in_view, tf.cast(1, tf.int32))

def static_func(data, keys):
    return tf.logical_not(data['is_moving'])

def moving_func(data, keys):
    return data['is_moving']

def is_teleporting_func(data, keys):
    return tf.logical_not(data['is_not_teleporting'])

def static_and_not_teleporting_func(data, keys):
    assert keys[0] in ['is_moving', 'is_moving2', 'is_moving3']
    assert keys[1] == 'is_not_teleporting'
    assert keys[2] == 'is_object_in_view'
    static = tf.logical_and(tf.logical_not(data[keys[0]]), data[keys[1]])
    return tf.logical_and(static, any_inview_func(data, keys[2:]))

def moving_and_not_teleporting_func(data, keys):
    assert keys[0] in ['is_moving', 'is_moving2', 'is_moving3']
    assert keys[1] == 'is_not_teleporting'
    assert keys[2] == 'is_object_in_view'
    moving = tf.logical_and(data[keys[0]], data[keys[1]])
    return tf.logical_and(moving, any_inview_func(data, keys[2:]))

def in_view_and_not_teleporting_func(data, keys):
    assert keys[0] == 'is_not_teleporting'
    assert keys[1] == 'is_object_in_view'
    return tf.logical_and(tf.logical_not(data[keys[0]]), any_inview_func(data, keys[1:]))

def shift_back_one(filter_bool):
    assert len(filter_bool.shape.as_list()) == 3

def moving_and_any_inview_and_not_acting_func(data, keys):
    assert (keys[0] in ['is_moving', 'is_moving2', 'is_moving3']) and (keys[1] == 'is_object_in_view'), keys
    is_moving_and_inview = moving_and_any_inview_func(data, keys[:2])
    assert 'is_acting' in keys[2:]
    not_acting = tf.logical_not(data['is_acting'])
    return tf.logical_and(not_acting, is_moving_and_inview)

def moving_and_any_inview_and_not_acting_and_not_teleporting_func(data, keys):
    assert (keys[0] in ['is_moving', 'is_moving2', 'is_moving3']) and (keys[1] == 'is_object_in_view'), keys
    is_moving_and_inview = moving_and_any_inview_func(data, keys[:2])
    assert 'is_acting' in keys[2:]
    assert 'is_not_teleporting' in keys[2:]
    not_acting = tf.logical_not(data['is_acting'])
    print('not acting shape', not_acting.shape.as_list())
    print("not teleporting_shape", data['is_not_teleporting'])
    return tf.logical_and(tf.logical_and(not_acting, is_moving_and_inview), data['is_not_teleporting'])

def not_teleporting_func(data, keys):
    assert 'is_not_teleporting' in keys
    return data['is_not_teleporting']

def not_acting_or_teleporting_func(data, keys):
    assert keys == ['is_acting', 'is_not_teleporting']
    return tf.logical_and(tf.logical_not(data['is_acting']), data['is_not_teleporting'])

def moving_and_any_inview_and_not_acting_and_max_particles_func(data, keys,
        max_num_particles):
    assert (keys[0] in ['is_moving', 'is_moving2', 'is_moving3']) and \
            (keys[1] == 'is_object_in_view') and \
            (keys[2] == 'is_acting') and \
            (keys[3] == 'num_particles'), keys

    valid_frames = moving_and_any_inview_and_not_acting_func(data, keys[:3])
    less_max_particles = tf.less(data['num_particles'], max_num_particles)
    return tf.logical_and(not_acting, less_max_particles)

def moving_and_any_inview_and_not_acting_and_128_particles_func(data, keys):
    return moving_and_any_inview_and_not_acting_and_max_particles_func(
            data, keys, 128)

def moving_and_any_inview_and_not_acting_and_256_particles_func(data, keys):
    return moving_and_any_inview_and_not_acting_and_max_particles_func(
            data, keys, 256)

def moving_and_any_inview_and_not_acting_and_512_particles_func(data, keys):
    return moving_and_any_inview_and_not_acting_and_max_particles_func(
            data, keys, 512)

def moving_and_all_inview_func(data, keys):
    assert all(k in keys for k in ['is_moving', 'is_object_in_view']), keys
    all_obj_in_view = tf.reduce_min(tf.cast(data['is_object_in_view'], tf.uint8), axis=2, keepdims=True)
    all_obj_in_view = tf.cast(all_obj_in_view, tf.bool)
    print('move_view_shape', data['is_moving'].shape, all_obj_in_view)
    return tf.logical_and(data['is_moving'], all_obj_in_view)

def moving_and_inview_func(data, keys):
    assert all(k in keys for k in ['is_moving', 'is_object_in_view']), keys
    print('move_view_shape', data['is_moving'].shape, data['is_object_in_view'].shape)
    return tf.logical_and(data['is_moving'], data['is_object_in_view'])

def moving_and_inview_and_valid_func(data, keys):
    assert all(k in keys for k in ['is_moving', 'is_object_in_view']), keys
    print('move_view_shape', data['is_moving'].shape, data['is_object_in_view'].shape)

    data_filter = tf.logical_and(data['is_moving'], data['is_object_in_view'])
    print("full_particles_agent_shape", data['full_particles_agent'].shape)
    valid_particle_inds = tf.cast(data['full_particles_agent'][:,:,:,14:15] != 0, dtype=tf.bool)
    print("valid_inds_shape", valid_particle_inds.shape)
    data_filter = tf.logical_and(data_filter, valid_particle_inds)
    return data_filter

class Sentinel(object):
    """
    Sentinel object
    """
    def __init__(self,
            name):
        assert isinstance(name, str)
        self.name = '<' + name + '>'

    def __repr__(self):
        return self.name

class Struct(object):
    """
    Converts data into struct accessible with "."
    """
    def __init__(self, data):
        for name, value in data.iteritems():
            setattr(self, name, self._wrap(value))


    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value


    def __repr__(self):
        return 'Struct({%s})' % str(', '.join("'%s': %s" % (k, repr(v)) \
                for (k, v) in self.__dict__.iteritems()))



class Filter(object):
    """
    Evaluates a logical expression where the symbols are keys to a dictionary
    of tensorflow tensors
    """
    str_to_token = {
            'and': lambda left, right: tf.logical_and(left, right),
            'or': lambda left, right: tf.logical_or(left, right),
            'not': lambda var: tf.logical_not(var),
            '(': '(',
            ')': ')',
            }
    empty_res = True


    def __init__(self, expression):
        """Either use a logical expression to initialize Filter or
        define a func and keys that takes the data dict and these filter keys
        as input: func(data, keys)"""
        if isinstance(expression, str):
            self.expression = expression
            self.token_lst = self.create_token_lst(self.expression)
        elif isinstance(expression, tuple):
            assert callable(expression[0]), expression[0]
            assert isinstance(expression[1], list), expression[1]
            self.func = expression[0]
            self.keys = expression[1]
            if len(expression) > 2:
                assert isinstance(expression[2], dict), expression[2]
                self.kwargs = expression[2]
            else:
                self.kwargs = {}
        else:
            raise Exception('Unknown initialization')


    def create_token_lst(self, expression, str_to_token=str_to_token):
        """create token list:
        'True or False' -> [True, lambda..., False]"""
        s = expression.replace('(', ' ( ')
        s = s.replace(')', ' ) ')

        token_lst = []
        self.keys = []
        for it in s.split():
            if it in str_to_token:
                token_lst.append(str_to_token[it])
            else:
                token_lst.append(it)
                self.keys.append(it)
        self.keys = np.unique(self.keys)
        return token_lst


    def find(self, lst, what, start=0):
        return [i for i, it in enumerate(lst) if it == what and i >= start]


    def parens(self, token_lst):
        """returns:
            (bool)parens_exist, left_paren_pos, right_paren_pos
        """
        left_lst = self.find(token_lst, '(')

        if not left_lst:
            return False, -1, -1

        left = left_lst[-1]

        #can not occur earlier, hence there are args and op.
        right = self.find(token_lst, ')', left + 1)[0]

        return True, left, right


    def bool_eval(self, token_lst, data):
        """token_lst has length 3 and format: [left_arg, operator, right_arg]
        operator(left_arg, right_arg) is returned"""
        try:
            if len(token_lst) == 2:
                assert callable(token_lst[0])
                operator = token_lst[0]
                if isinstance(token_lst[1], str):
                    var = tf.cast(data[token_lst[1]], tf.bool)
                else:
                    var = token_lst[1]
                return operator(var)
            else:
                assert len(token_lst) == 3
                assert callable(token_lst[1])
                operator = token_lst[1]
                if isinstance(token_lst[0], str):
                    lhs = tf.cast(data[token_lst[0]], tf.bool)
                else:
                    lhs = token_lst[0]
                if isinstance(token_lst[2], str):
                    rhs = tf.cast(data[token_lst[2]], tf.bool)
                else:
                    rhs = token_lst[2]

                return operator(lhs, rhs)
        except AssertionError:
            raise AssertionError('Every expression has to be ' + \
                    'encapsulated in brackets as a 3-tuple:\n' + \
                    '(left_arg operator right_arg)\n' + \
                    '\'not\' has to be written as:\n' + \
                    '(not arg)\n' + \
                    'Your given expression: %s' % self.expression)


    def formatted_bool_eval(self, token_lst, data, empty_res=empty_res):
        """eval a formatted (i.e. of the form 'ToFa(ToF)') string"""
        if not token_lst:
            return self.empty_res

        if len(token_lst) == 1:
            if isinstance(token_lst[0], str):
                return data[token_lst[0]]
            else:
                return token_lst[0]

        has_parens, l_paren, r_paren = self.parens(token_lst)

        if not has_parens:
            return self.bool_eval(token_lst, data)

        token_lst[l_paren:r_paren + 1] = [self.bool_eval(
            token_lst[l_paren+1:r_paren], data)]

        return self.formatted_bool_eval(token_lst, data, self.bool_eval)


    def eval(self, data):
        """The actual 'eval' routine,
        if 's' is empty, 'True' is returned,
        otherwise 's' is evaluated according to parentheses nesting."""
        self.data = data
        if hasattr(self, 'func'):
            return self.func(self.data, self.keys, **self.kwargs)
        else:
            return self.formatted_bool_eval(
                    self.token_lst, self.data)


def filter_tests():
    sess = tf.Session()
    data = {'a': [1, 0], 'b': [0, 1], 'c': [1, 0]}
    #data = {'a': 1, 'b': 0, 'c': 1}
    # Test and
    expr = 'a and b'
    f1 = Filter(expr)
    assert (sess.run(f1.eval(data)) == [False, False]).all(), '%s != %s' % \
            (expr, [False, False])
    # Test or
    expr = 'a or b'
    f2 = Filter(expr)
    assert (sess.run(f2.eval(data)) == [True, True]).all(), '%s != %s' % \
            (expr, [True, True])
    # Test chain
    expr = '(a and b) and c'
    f3 = Filter(expr)
    assert (sess.run(f3.eval(data)) == [False, False]).all(), '%s != %s' % \
            (expr, [False, False])
    # Test brackets
    expr = 'a and (c and b)'
    f4 = Filter(expr)
    assert (sess.run(f4.eval(data)) == [False, False]).all(), '%s != %s' % \
            (expr, [False, False])
    # Test nested brackets
    expr = '((a and b) or (a and (a or b)))'
    f5 = Filter(expr)
    assert (sess.run(f5.eval(data)) == [True, False]).all(), '%s != %s' % \
            (expr, [True, False])
    # Test not
    expr = '(not (b or (not (b or a))))'
    f6 = Filter(expr)
    assert (sess.run(f6.eval(data)) == [True, False]).all(), '%s != %s' % \
            (expr, [True, False])
    # Test func
    f = Filter((lambda data, keys: [data[key] for key in keys], ['a', 'b', 'c']))
    assert f.eval(data) == [data[key] for key in ['a', 'b', 'c']], '%s != %s' % \
            (f.eval(data), [data[key] for key in ['a', 'b', 'c']])

    print('FILTER TESTS PASSED')


if __name__ == '__main__':
    filter_tests()
