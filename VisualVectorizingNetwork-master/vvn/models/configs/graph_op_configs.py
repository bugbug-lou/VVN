import tensorflow as tf
import numpy as np
import copy

from graph.common import Graph, propdict, graph_op, node_op, edge_op
import graph.compute as gcompute
import graph.build as gbuild

from vvn.ops.convolutional import mlp
from vvn.ops.utils import rotate_vectors_with_quaternions, preproc_hsv
from vvn.ops.dimensions import DimensionDict, OrderedDict

from vvn.models.dynamics import PhysicsModel


def surface_areas_from_attrs(nodes, f=1.95, near_plane=0.5, eps=1e-2):
    z = tf.maximum(-nodes['positions'][...,2:3], near_plane)
    px_areas = nodes['areas']
    nz = tf.abs(nodes['normals'][...,2:3] + eps)
    areas = tf.square(z / f) * px_areas
    areas = areas / nz
    return areas

def interpolate_velocities(nodes):
    prev_vels = nodes['prev_velocities']
    vels = nodes['velocities']
    gate = nodes['velocities_gate']
    valid = nodes['matches_valid']

    full_gate = (1 - tf.nn.sigmoid(gate)) * valid # must be using a valid match to update
    next_vels = full_gate * prev_vels + (1 - full_gate) * vels
    return next_vels

def vz_scale(nodes, multiplier=1.0):
    z = tf.maximum(-nodes['positions'][...,2:3], 0.2)
    vz = -nodes['velocities'][...,2:3]
    return tf.minimum(tf.maximum(((multiplier*vz) + z) / z, 0.0), 3.0)

scaling_ops = [
    {
        'func': vz_scale,
        'op_type': 'node',
        'node_key': 'nodes_level_1',
        'node_feat_keys': ['positions', 'velocities'],
        'output_node_feat_key': 'scale_modifier',
        'multiplier': 1.0
    },
    {
        'func': tf.multiply,
        'op_type': 'transform',
        'node_key': 'nodes_level_1',
        'attr_key': 'scales_2d',
        'effects_key': 'scale_modifier'
    }
]

init_scales = {'nodes_level_1':
    {'scales_2d': 1.0}
}

amplify_velocity = [
    {
        'func': lambda n: 0.0 * n['velocities'],
        'op_type': 'node',
        'node_key': 'nodes_level_1',
        'node_feat_keys': ['velocities'],
        'output_node_feat_key': 'velocities'
    }
    # {
    #     'func': lambda effs, vels: 0.0 * vels,
    #     'op_type': 'transform',
    #     'node_key': 'nodes_level_1',
    #     'attr_key': ['velocities'],
    #     'effects_key': 'velocities'
    # }
]

set_velocity = [
    {
        'func': lambda n: -tf.ones_like(n['velocities']),
        'op_type': 'node',
        'node_key': 'nodes_level_1',
        'node_feat_keys': ['velocities'],
        'output_node_feat_key': 'velocities'
    }
]

delta_ops = [
    # estimate velocities from deltas
    {
        'func': PhysicsModel.features_from_attrs,
        'op_type': 'node',
        'node_key': 'nodes_level_1',
        # 'node_feat_keys': ['delta_positions', 'cost', 'velocities', 'volumes'],
        'node_feat_keys': ['delta_positions', 'cost', 'velocities', 'volumes', 'positions', 'areas', 'velocities_mse'],
        'output_node_feat_key': 'prev_velocities',
        'node_dims': [50,50,3],
        'activation': tf.nn.relu
    },
    # gate velocities update by whether it was a real match
    {
        'func': PhysicsModel.features_from_attrs,
        'op_type': 'node',
        'node_key': 'nodes_level_1',
        # 'node_feat_keys': ['delta_positions', 'cost', 'velocities', 'volumes'],
        'node_feat_keys': ['delta_positions', 'cost', 'velocities', 'volumes', 'positions', 'areas', 'velocities_mse'],
        'output_node_feat_key': 'velocities_gate',
        'node_dims': [50,50,1],
        'activation': tf.nn.relu
    },
    # interpolate
    {
        'func': interpolate_velocities,
        'op_type': 'node',
        'node_key': 'nodes_level_1',
        'node_feat_keys': ['prev_velocities', 'velocities', 'velocities_gate', 'matches_valid'],
        'output_node_feat_key': 'next_velocities'
    },
    # update
    {
        'func': lambda effects, attrs: effects,
        'op_type': 'transform',
        'node_key': 'nodes_level_1',
        'attr_key': ['velocities', 'delta_positions'],
        'effects_key': 'next_velocities',
        'reduce_dim': False
    }
]



soft_collide_ops = [
    {
        'func': surface_areas_from_attrs,
        'op_type': 'node',
        'node_key': 'nodes_level_1',
        'node_feat_keys': ['positions', 'areas', 'normals'],
        'output_node_feat_key': 'surface_areas'
    },
    {
        'func': PhysicsModel.graphconv_from_attrs,
        'op_type': 'graph',
        'edge_key': 'all_to_all_edges_1_to_1',
        'sender_feat_keys': ['positions', 'velocities', 'normals', 'areas', 'surface_areas', 'shapes', 'normals_mse', 'spatial_normals', 'borders', 'velocities_mse', 'volumes'],
        'receiver_feat_keys': ['positions', 'velocities', 'normals', 'areas', 'surface_areas', 'shapes', 'normals_mse', 'spatial_normals', 'borders', 'velocities_mse', 'volumes'],
        'diff_attrs': ['positions', 'velocities', 'normals'],
        'diff_funcs': {'normals': lambda x,y: tf.reduce_sum(x*y, axis=-1, keepdims=True)},
        'output_receiver_feat_key': 'collision_effects',
        'node_dims': [200, 200, 200, 20],
        'activation': tf.nn.elu
    },
    # {
    #     'func': PhysicsModel.features_from_attrs,
    #     'op_type': 'node',
    #     'node_key': 'nodes_level_1',
    #     'node_feat_keys': ['collision_effects'],
    #     'output_node_feat_key': 'velocity_quaternions',
    #     'node_dims': [4],
    #     'activation': None
    # },
    # {
    #     'func': rotate_vectors_with_quaternions,
    #     'op_type': 'transform',
    #     'node_key': 'nodes_level_1',
    #     'attr_key': 'velocities',
    #     'effects_key': 'velocity_quaternions'
    # },
    {
        'func': tf.add,
        'op_type': 'transform',
        'node_key': 'nodes_level_1',
        'attr_key': ['velocities'],
        'effects_key': 'collision_effects',
        'reduce_dim': True
    }
]

delta_ops_mlp = [
    # estimate velocities from deltas
    {
        'func': lambda nodes: nodes['delta_positions'],
        'op_type': 'node',
        'node_key': 'nodes_level_1',
        'node_feat_keys': ['delta_positions'],
        'output_node_feat_key': 'prev_velocities'
    },
    # gate velocities update by whether it was a real match
    {
        'func': PhysicsModel.features_from_attrs,
        'op_type': 'node',
        'node_key': 'nodes_level_1',
        'node_feat_keys': ['delta_positions', 'cost', 'velocities', 'velocities_mse', 'volumes'],
        'output_node_feat_key': 'velocities_gate',
        'node_dims': [50,50,1],
        'activation': tf.nn.relu
    },
    # mlp to predict new velocities + overwrite
    {
        'func': PhysicsModel.features_from_attrs,
        'op_type': 'node',
        'node_key': 'nodes_level_1',
        'node_feat_keys': ['delta_positions', 'velocities', 'velocities_mse', 'volumes', 'shapes', 'normals', 'normals_mse', 'spatial_depths', 'spatial_normals'],
        'output_node_feat_key': 'next_velocities',
        'node_dims': [50,50,3],
        'activation': tf.nn.relu
    },
    {
        'func': lambda effects, attrs: effects,
        'op_type': 'transform',
        'node_key': 'nodes_level_1',
        'attr_key': ['velocities'],
        'effects_key': 'next_velocities',
        'reduce_dim': False
    },
    # interpolate
    {
        'func': interpolate_velocities,
        'op_type': 'node',
        'node_key': 'nodes_level_1',
        'node_feat_keys': ['prev_velocities', 'velocities', 'velocities_gate', 'matches_valid'],
        'output_node_feat_key': 'next_velocities'
    },
    # update
    {
        'func': lambda effects, attrs: effects,
        'op_type': 'transform',
        'node_key': 'nodes_level_1',
        'attr_key': ['velocities', 'delta_positions'],
        'effects_key': 'next_velocities',
        'reduce_dim': False
    }
]

push_forward_ops = [
    {
        'func': PhysicsModel.project_velocity_hw,
        'op_type': 'node',
        'node_key': 'nodes_level_1',
        'node_feat_keys': ['positions', 'velocities'],
        'near_plane': -0.2,
        'output_node_feat_key': 'hw_velocities',
        'use_pmat': True,
    },
    {
        'func': lambda effs, hw: tf.clip_by_value(effs + hw, -10.0, 10.0),
        'op_type': 'transform',
        'node_key': 'nodes_level_1',
        'attr_key': 'hw_centroids',
        'effects_key': 'hw_velocities'
    },
    {
        'func': lambda vs, xs: xs + tf.clip_by_value(vs, -100.0, 100.0),
        'op_type': 'transform',
        'node_key': 'nodes_level_1',
        'attr_key': 'positions',
        'effects_key': 'velocities'
    },
    {
        'func': PhysicsModel.scatter_effects,
        'op_type': 'graph',
        'edge_key': 'parent_to_child_edges_1_to_0',
        'sender_feat_keys': ['velocities'],
        'receiver_feat_keys': ['valid'],
        'output_receiver_feat_key': 'parent_velocities'
    },
    {
        'func': PhysicsModel.scatter_effects,
        'op_type': 'graph',
        'edge_key': 'parent_to_child_edges_1_to_0',
        'sender_feat_keys': ['hw_velocities'],
        'receiver_feat_keys': ['valid'],
        'output_receiver_feat_key': 'hw_velocities'
    },
    {
        'func': lambda effs, hw: tf.clip_by_value(effs + hw, -10.0, 10.0),
        'op_type': 'transform',
        'node_key': 'nodes_level_0',
        'attr_key': 'hw_centroids',
        'effects_key': 'hw_velocities'
    },
    {
        'func': lambda vs, xs: xs + tf.clip_by_value(vs, -10.0, 10.0),
        'op_type': 'transform',
        'node_key': 'nodes_level_0',
        'attr_key': 'positions',
        'effects_key': 'velocities'
    }
]

push_forward_top_ops = push_forward_ops[:3]

damping_ops = [
    {
        'func': PhysicsModel.interacting_edges_by_distance,
        'op_type': 'graph',
        'edge_key': 'across_parent_edges_0_to_0',
        'sender_feat_keys': ['positions'],
        'receiver_feat_keys': ['positions'],
        'output_receiver_feat_key': None,
        'output_edge_feat_key': 'interacting',
        'distance_thresh': 1.0,
        'binarize': True
    },
    {
        'func': PhysicsModel.collide_interacting_nodes,
        'op_type': 'graph',
        'edge_key': 'across_parent_edges_0_to_0',
        'edge_feat_keys': ['interacting'],
        'sender_feat_keys': ['positions', 'velocities', 'normals', 'relative_positions'],
        'receiver_feat_keys': ['positions', 'velocities', 'normals', 'relative_positions'],
        'diff_attrs': ['positions', 'normals', 'velocities'],
        'diff_funcs': {'normals': lambda x,y: tf.reduce_sum(x*y, axis=-1, keepdims=True)},
        'output_receiver_feat_key': 'damping_effects',
        'node_dims': [50,20],
        'activation': tf.nn.elu
    },
    {
        'func': lambda effects, vels: tf.nn.sigmoid(effects)*vels,
        'op_type': 'transform',
        'node_key': 'nodes_level_0',
        'attr_key': ['velocities'],
        'effects_key': 'damping_effects',
        'reduce_dim': True
    },
    {
        'func': PhysicsModel.graphconv_from_attrs,
        'op_type': 'graph',
        'edge_key': 'child_to_parent_edges_0_to_1',
        'sender_feat_keys': ['damping_effects'],
        # 'receiver_feat_keys': ['normals', 'velocities', 'volumes', 'shapes', 'normals_mse', 'velocities_mse', 'spatial_depths', 'spatial_normals'],
        'receiver_feat_keys': ['normals', 'velocities', 'volumes', 'shapes', 'normals_mse', 'velocities_mse', 'spatial_depths', 'spatial_normals', 'areas', 'surface_areas'],
        'output_receiver_feat_key': 'damping_effects',
        'node_dims': [20],
        'activation': tf.nn.elu
    },
    {
        'func': lambda effects, vels: tf.nn.sigmoid(effects)*vels,
        'op_type': 'transform',
        'node_key': 'nodes_level_1',
        'attr_key': ['velocities'],
        'effects_key': 'damping_effects',
        'reduce_dim': True
    }
]


relative_pos_ops = [
    {
        'func': PhysicsModel.scatter_effects,
        'op_type': 'graph',
        'edge_key': 'parent_to_child_edges_1_to_0',
        'sender_feat_keys': ['positions'],
        'receiver_feat_keys': ['valid'],
        'output_receiver_feat_key': 'parent_positions'
    },
    {
        'func': lambda nodes: nodes['positions'] - nodes['parent_positions'],
        'op_type': 'node',
        'node_key': 'nodes_level_0',
        'node_feat_keys': ['positions', 'parent_positions'],
        'output_node_feat_key': 'relative_positions'
    }
]

collide_top_ops = [
    {
        'func': surface_areas_from_attrs,
        'op_type': 'node',
        'node_key': 'nodes_level_1',
        'node_feat_keys': ['positions', 'areas', 'normals'],
        'output_node_feat_key': 'surface_areas'
    },
    {
        'func': PhysicsModel.interacting_edges_by_distance,
        'op_type': 'graph',
        'edge_key': 'all_to_all_edges_1_to_1',
        'sender_feat_keys': ['positions', 'inview'],
        'receiver_feat_keys': ['positions', 'inview'],
        'output_edge_feat_key': 'interacting',
        'output_receiver_feat_key': None,
        'distance_thresh': 1.0,
        'valid_key': 'valid',
        'binarize': True
    },
    {
        'func': PhysicsModel.collide_interacting_nodes,
        'op_type': 'graph',
        'edge_key': 'all_to_all_edges_1_to_1',
        'edge_feat_keys': ['interacting'],
        # 'sender_feat_keys': ['positions', 'velocities', 'normals', 'surface_areas', 'shapes'],
        # 'receiver_feat_keys': ['positions', 'velocities', 'normals', 'surface_areas', 'shapes'],
        # 'sender_feat_keys': ['positions', 'velocities', 'normals', 'surface_areas', 'shapes', 'positions_backward_euler', 'normals_backward_euler', 'positions_var', 'normals_var', 'positions_hmoment', 'positions_wmoment', 'normals_hmoment', 'normals_wmoment'],
        # 'receiver_feat_keys': ['positions', 'velocities', 'normals', 'surface_areas', 'shapes', 'positions_backward_euler', 'normals_backward_euler', 'positions_var', 'normals_var', 'positions_hmoment', 'positions_wmoment', 'normals_hmoment', 'normals_wmoment'],
        'sender_feat_keys': ['positions', 'velocities', 'normals', 'surface_areas', 'shapes', 'positions_backward_euler', 'normals_backward_euler', 'positions_var', 'normals_var', 'positions_hmoment', 'positions_wmoment', 'normals_hmoment', 'normals_wmoment', 'positions_prev1', 'positions_prev2', 'positions_prev3', 'normals_prev1', 'normals_prev2', 'normals_prev3'],
        'receiver_feat_keys': ['positions', 'velocities', 'normals', 'surface_areas', 'shapes', 'positions_backward_euler', 'normals_backward_euler', 'positions_var', 'normals_var', 'positions_hmoment', 'positions_wmoment', 'normals_hmoment', 'normals_wmoment', 'positions_prev1', 'positions_prev2', 'positions_prev3', 'normals_prev1', 'normals_prev2', 'normals_prev3'],
        'diff_attrs': ['positions', 'normals', 'velocities'],
        'diff_funcs': {'normals': lambda x,y: tf.reduce_sum(x*y, axis=-1, keepdims=True)},
        'output_receiver_feat_key': 'collision_effects',
        # 'node_dims': [50,50],
        'node_dims': [250,250],
        'activation': tf.nn.elu
    },
    {
        'func': PhysicsModel.features_from_attrs,
        'op_type': 'node',
        'node_key': 'nodes_level_1',
        # 'node_feat_keys': ['velocities', 'positions', 'normals', 'areas', 'shapes', 'normals', 'collision_effects'],
        'node_feat_keys': ['positions', 'velocities', 'normals', 'areas', 'surface_areas', 'shapes', 'positions_backward_euler', 'normals_backward_euler', 'positions_var', 'normals_var', 'positions_hmoment', 'positions_wmoment', 'normals_hmoment', 'normals_wmoment', 'collision_effects'],
        'output_node_feat_key': 'collision_effects',
        # 'node_dims': [50,50,3],
        'node_dims': [100,100,3],
        'activation': tf.nn.elu
    },
    {
        # 'func': lambda effects, vels: tf.log(1.0 + tf.nn.relu(effects))*vels,
        # 'func': lambda effects, vels: vels * (1.0 + tf.tanh(effects)),
        # 'func': lambda effects, vels: tf.log(1.0 + tf.nn.relu(effects))*vels,
        # 'func': lambda effects, vels: vels * (1.0 + tf.tanh(effects)),
        # 'func': lambda effects, vels: vels + (tf.log(1.0 + tf.abs(effects))*tf.sign(effects)),
        # 'func': lambda effects, vels: vels + (tf.log(1.0 + tf.square(effects))*tf.sign(effects)),
        'func': lambda effects, vels: vels + effects,

        'op_type': 'transform',
        'node_key': 'nodes_level_1',
        'attr_key': ['velocities'],
        'effects_key': 'collision_effects',
        'reduce_dim': True
    }
]

collide_shapepres_ops = [
    {
        'func': surface_areas_from_attrs,
        'op_type': 'node',
        'node_key': 'nodes_level_1',
        'node_feat_keys': ['positions', 'areas', 'normals'],
        'output_node_feat_key': 'surface_areas'
    },
    {
        'func': PhysicsModel.scatter_effects,
        'op_type': 'graph',
        'edge_key': 'parent_to_child_edges_1_to_0',
        'sender_feat_keys': ['positions'],
        'receiver_feat_keys': ['valid'],
        'output_receiver_feat_key': 'parent_positions'
    },
    {
        'func': lambda nodes: nodes['positions'] - nodes['parent_positions'],
        'op_type': 'node',
        'node_key': 'nodes_level_0',
        'node_feat_keys': ['positions', 'parent_positions'],
        'output_node_feat_key': 'relative_positions'
    },
    {
        'func': PhysicsModel.interacting_edges_by_distance,
        'op_type': 'graph',
        'edge_key': 'across_parent_edges_0_to_0',
        'sender_feat_keys': ['positions', 'valid'],
        'receiver_feat_keys': ['positions', 'valid'],
        'output_edge_feat_key': 'interacting',
        'output_receiver_feat_key': None,
        'distance_thresh': 0.1,
        'valid_key': 'valid',
        'binarize': True
    },
    {
        'func': PhysicsModel.collide_interacting_nodes,
        'op_type': 'graph',
        'edge_key': 'across_parent_edges_0_to_0',
        'edge_feat_keys': ['interacting'],
        'sender_feat_keys': ['positions', 'velocities', 'normals', 'relative_positions'],
        'receiver_feat_keys': ['positions', 'velocities', 'normals', 'relative_positions'],
        'diff_attrs': ['positions', 'normals', 'velocities'],
        'diff_funcs': {'normals': lambda x,y: tf.reduce_sum(x*y, axis=-1, keepdims=True)},
        'output_receiver_feat_key': 'collision_effects',
        'node_dims': [50,50],
        'activation': tf.nn.elu
    },
    {
        'func': PhysicsModel.graphconv_from_attrs,
        'op_type': 'graph',
        'edge_key': 'child_to_parent_edges_0_to_1',
        'sender_feat_keys': ['collision_effects'],
        # 'receiver_feat_keys': ['normals', 'velocities', 'volumes', 'shapes', 'normals_mse', 'velocities_mse', 'spatial_depths', 'spatial_normals'],
        # 'receiver_feat_keys': ['normals', 'velocities', 'volumes', 'shapes', 'normals_mse', 'velocities_mse', 'spatial_depths', 'spatial_normals', 'areas', 'surface_areas'],
        'receiver_feat_keys': ['normals', 'velocities', 'shapes', 'areas', 'surface_areas'],
        'output_receiver_feat_key': 'collision_effects',
        'node_dims': [100,100,20],
        'activation': tf.nn.elu
    },
    {
        # 'func': lambda effects, vels: tf.log(1.0 + tf.nn.relu(effects))*vels,
        # 'func': lambda effects, vels: vels * (1.0 + tf.tanh(effects)),
        # 'func': lambda effects, vels: tf.log(1.0 + tf.nn.relu(effects))*vels,
        # 'func': lambda effects, vels: vels * (1.0 + tf.tanh(effects)),
        # 'func': lambda effects, vels: vels + (tf.log(1.0 + tf.abs(effects))*tf.sign(effects)),
        'func': lambda effects, vels: vels + (tf.log(1.0 + tf.square(effects))*tf.sign(effects)),
        'op_type': 'transform',
        'node_key': 'nodes_level_1',
        'attr_key': ['velocities'],
        'effects_key': 'collision_effects',
        'reduce_dim': True
    }
]

rotate_ops = [
    {
        'func': PhysicsModel.features_from_attrs,
        'op_type': 'node',
        'node_key': 'nodes_level_1',
        'node_feat_keys': ['collision_effects'],
        'output_node_feat_key': 'quaternions',
        'node_dims': [4],
        'activation': None
    },
    {
        'func': rotate_vectors_with_quaternions,
        'op_type': 'transform',
        'node_key': 'nodes_level_1',
        'attr_key': ['normals', 'volumes'],
        'effects_key': 'quaternions'
    },
    {
        'func': PhysicsModel.scatter_effects,
        'op_type': 'graph',
        'edge_key': 'parent_to_child_edges_1_to_0',
        'sender_feat_keys': ['quaternions'],
        'receiver_feat_keys': ['valid'],
        'output_receiver_feat_key': 'quaternions'
    },
    {
        'func': rotate_vectors_with_quaternions,
        'op_type': 'transform',
        'node_key': 'nodes_level_0',
        'attr_key': ['relative_positions', 'normals'],
        'effects_key': 'quaternions'
    },
    {
        'func': lambda nodes: nodes['relative_positions'] + nodes['parent_positions'],
        'op_type': 'node',
        'node_key': 'nodes_level_0',
        'node_feat_keys': ['relative_positions', 'parent_positions'],
        'output_node_feat_key': 'positions'
    },
    {
        'func': lambda effs, attr: effs,
        'op_type': 'transform',
        'node_key': 'nodes_level_0',
        'attr_key': 'positions',
        'effects_key': 'positions'
    }
]

self_rotate_ops = copy.deepcopy(rotate_ops)[1:]
self_rotate_ops = [
    {
        'func': PhysicsModel.scatter_effects,
        'op_type': 'graph',
        'edge_key': 'parent_to_child_edges_1_to_0',
        'sender_feat_keys': ['positions'],
        'receiver_feat_keys': ['valid'],
        'output_receiver_feat_key': 'parent_positions'
    },
    {
        'func': lambda nodes: nodes['positions'] - nodes['parent_positions'],
        'op_type': 'node',
        'node_key': 'nodes_level_0',
        'node_feat_keys': ['positions', 'parent_positions'],
        'output_node_feat_key': 'relative_positions'
    },
    {
        'func': PhysicsModel.features_from_attrs,
        'op_type': 'node',
        'node_key': 'nodes_level_1',
        'node_feat_keys': ['positions', 'normals', 'volumes', 'velocities', 'shapes', 'borders'],
        'output_node_feat_key': 'quaternions',
        'node_dims': [50, 4],
        'activation': tf.nn.elu
    }
] + self_rotate_ops
