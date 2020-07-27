import numpy as np
import tensorflow as tf
import copy

# Imports from Physics repo
from physics.models.constructor.base import GraphConstructor
from graph.common import Graph, propdict, graph_op, node_op, edge_op
import graph.compute as gcompute
import graph.build as gbuild

# Imports from VVN
import vvn.ops.graphical as graphical
from vvn.models.rendering import agent_particles_to_image_coordinates
from vvn.ops.convolutional import mlp
from vvn.ops.dimensions import DimensionDict, OrderedDict

# for debugging training
PRINT = False

class HierarchicalVisualGraphConstructor(GraphConstructor):

    def __init__(self,
                 num_levels=2,
                 dimension_dicts=None,
                 attribute_dims=None,
                 attribute_preprocs=None,
                 stop_gradient_on_attributes=[],
                 build_object_nodes=False,
                 **kwargs
    ):

        self.num_levels = num_levels
        self.build_object_nodes = build_object_nodes

        # set the attribute dims
        if attribute_dims is None:
            attr_dims = {
                # 'positions': [0,3],
                # 'normals': [6,9],
                # 'velocities': [16,19],
                # 'hw_centroids': [-4,-2],
                # 'areas': [-2,-1],
                'valid':[-1,0],
                'vector':[0,0]
            } # default conventions
            self.attribute_dims = [attr_dims] * self.num_levels
        elif isinstance(attribute_dims, dict):
            attribute_dims.update({'valid':[-1,0], 'vector':[0,0]})
            self.attribute_dims = [attribute_dims] * self.num_levels
        else:
            assert isinstance(attribute_dims, list) and isinstance(attribute_dims[0], dict) and len(attribute_dims) == self.num_levels, "Must pass one dict per graph level"
            for A in attribute_dims:
                A.update({'valid':[-1,0], 'vector':[0,0]})
            self.attribute_dims = attribute_dims

        # set the attribute preprocs
        if attribute_preprocs is None:
            self.attribute_preprocs = [{'valid': lambda v: tf.cast(v > 0.5, dtype=tf.float32)}] * self.num_levels
        elif isinstance(attribute_preprocs, dict):
            # self.attribute_preprocs = [attribute_preprocs] * self.num_levels
            self.attribute_preprocs = [attribute_preprocs] * self.num_levels
        else:
            assert isinstance(attribute_preprocs, list) and isinstance(attribute_preprocs[0], dict) and len(attribute_preprocs) == self.num_levels
            self.attribute_preprocs = [copy.deepcopy(attribute_preprocs[level]) for level in range(self.num_levels)]

        # if dims were passed in
        if dimension_dicts is not None:
            assert len(dimension_dicts) == self.num_levels
            self.Dims = list(dimension_dicts.values())
            for level,D in enumerate(self.Dims):
                D.update({k:dims for k,dims in self.attribute_dims[level].items() if k not in D.keys()})
                D.update({k:func for k,func in self.attribute_preprocs[level].items()})

            self.attribute_dims = self.Dims
            self.attribute_preprocs = [{k:dims[2] for k,dims in D.items()} for D in self.Dims]

        else:
            self.Dims = None

        # stop gradient from flowing into some attributes
        self.stop_gradient_on_attributes = stop_gradient_on_attributes

    def vectors_to_attrs(self, node_vectors, attribute_dims, attribute_preprocs={}):
        '''
        Converts nodes in vector format to attribute format (i.e. a dict of attributes)

        node_vectors: [B,T,N,D] <tf.float32>
        attribute_dims: <dict> of (attr, [d0,d1]) pairs. By convention, d1 == 0 sets d1 <- D
        attribute_preprocs: <dict> of (attr, func) pairs to apply to each attr.

        returns:
        attrs: <dict> of (attr, [B,T,N, attribute_dims[attr][1] - attribute_dims[attr][0]] <tf.float32>) processed attrs
        '''
        B,T,N,D = node_vectors.shape.as_list()
        attr_outputs = {}
        attr_dims = {}
        for attr, dims in attribute_dims.items():
            d0, d1 = [(D+d if d<0 else d) for d in dims]
            d1 = D if d1==0 else d1
            func = attribute_preprocs.get(attr, tf.identity) or tf.identity
            attr_func = lambda tensor: tf.identity(func(tensor), name=(attr+'_preproc'))
            attr_out = attr_func(node_vectors[...,d0:d1])
            if attr in self.stop_gradient_on_attributes:
                print("stopping gradient on %s" % attr)
                attr_out = tf.stop_gradient(attr_out)
            attr_outputs[attr] = attr_out
            attr_dims[attr] = [d0, d1]

        return propdict(attr_outputs), attr_dims

    def update_vector(self, node_key, attr_key):
        '''
        Update the 'vector' attribute of self.graph[node_key]
        '''
        assert isinstance(self.attribute_dims, dict), "Can't update until the graph has been constructed"
        vectors = self.graph[node_key].get('vector', None)
        assert vectors is not None, "nodes must have a vectors attribute"
        attr_val = self.graph[node_key][attr_key]
        attr_dims = self.attribute_dims[node_key][attr_key]
        D = vectors.shape.as_list()[-1]
        d0, d1 = [(D+d if d<0 else d) for d in attr_dims]
        d1 = D if d1==0 else d1
        assert attr_val.shape.as_list()[-1] == (d1-d0)

        #update
        self.graph[node_key]['vector'] = tf.concat([
            vectors[...,:d0],
            attr_val,
            vectors[...,d1:]
        ], axis=-1)

    @staticmethod
    def add_object_level(nodes, **kwargs):
        pass

    @staticmethod
    def all_to_all_edges_from_nodes(graph, edge_key, node_key=None, self_edges=False):
        '''
        Adds 'all_to_all' edges to a graph at a level specified by the edge or node key
        '''
        if node_key is None:
            assert 'all_to_all' in edge_key
            level = edge_key.split('_')[-1] # str(int) of level
            node_key = 'nodes_level_%s' % level
        assert node_key in graph.nodes.keys()

        valid_nodes = graph.nodes[node_key]['valid'] # [B,T,N,1]
        B,T,N,_ = valid_nodes.shape.as_list()
        valid_edges = (valid_nodes * tf.transpose(valid_nodes, [0,1,3,2])) > 0.5 # bool

        if not self_edges:
            valid_edges = tf.logical_and(valid_edges, tf.logical_not(tf.eye(N, batch_shape=[B,T], dtype=tf.bool)))

        valid_edges = tf.cast(tf.where(valid_edges), tf.int32) # [?,4]

        level = node_key.split('_')[-1]
        edge_key = 'all_to_all_edges_%s_to_%s' % (level, level)
        graph.edges[edge_key] = propdict({
            'idx': valid_edges,
            'layer': np.array([node_key, node_key]).reshape([1,2])})
        return graph

    @staticmethod
    def across_parent_edges_from_nodes(graph, edge_key):

        assert 'across' in edge_key
        lev1, lev2 = edge_key.split('_')[-3::2]
        assert lev1 == lev2, "edge key must be connecting same level but levels are %s, %s" % (lev1, lev2)
        level = lev1
        node_key = 'nodes_level_%s' % level
        assert node_key in graph.nodes.keys()
        c2p_key = 'child_to_parent_edges_%s_to_%s' % (level, str(int(level)+1))
        assert c2p_key in graph.edges.keys(), "c2p_key: %s" % (c2p_key)
        c2p_edges = graph.edges[c2p_key]['idx'] # [?,4] inds
        valid_nodes = tf.cast(graph.nodes[node_key]['valid'], tf.bool) # [B,T,N,1]
        parents = tf.scatter_nd(indices=c2p_edges[:,0:3], updates=c2p_edges[:,3:4], shape=valid_nodes.shape) # [B,T,N,1]
        diff_parents = tf.logical_not(tf.equal(parents, tf.transpose(parents, [0,1,3,2]))) # [B,T,N,N]
        diff_parents = tf.logical_and(diff_parents, tf.logical_and(valid_nodes, tf.transpose(valid_nodes, [0,1,3,2]))) # [B,T,N,N]

        graph.edges[edge_key] = propdict({
            'idx': tf.cast(tf.where(diff_parents), tf.int32),
            'layer': np.array([node_key, node_key]).reshape([1,2])})
        return graph

    @staticmethod
    def within_parent_edges_from_nodes(graph, edge_key, self_edges=False):

        assert 'within_parent' in edge_key
        lev1, lev2 = edge_key.split('_')[-3::2]
        assert lev1 == lev2
        level = lev1
        c2p_key = 'child_to_parent_edges_%s_to_%s' % (level, str(int(level)+1))
        assert c2p_key in graph.edges.keys()

        c2p_edges = graph.edges[c2p_key]['idx'] # [?,4]
        node_key = graph.edges[c2p_key]['layer'].squeeze()[0]
        valid_nodes = tf.cast(graph.nodes[node_key]['valid'], tf.bool) # [B,T,N,1]
        B,T,N,_ = valid_nodes.shape.as_list()
        parents = tf.scatter_nd(indices=c2p_edges[:,0:3], updates=c2p_edges[:,3:4], shape=valid_nodes.shape) # [B,T,N,1]
        same_parents = tf.equal(parents, tf.transpose(parents, [0,1,3,2]))
        if not self_edges:
            same_parents = tf.logical_and(same_parents, tf.logical_not(tf.eye(N, batch_shape=[B,T], dtype=tf.bool)))
        same_parents = tf.logical_and(same_parents, tf.logical_and(valid_nodes, tf.transpose(valid_nodes, [0,1,3,2]))) # [B,T,N,N]

        graph.edges[edge_key] = propdict({
            'idx': tf.cast(tf.where(same_parents), tf.int32),
            'layer': np.array([node_key, node_key]).reshape([1,2])
        })
        return graph

    @staticmethod
    def copy_graph(graph):

        nodes = graph.nodes
        nodes_copy = {
            nk: propdict({
                ak: tf.identity(nodes[nk][ak], name=('%s_copy'%ak)) for ak in nodes[nk].keys()
            }) for nk in nodes.keys()
        }

        edges = graph.edges
        edges_copy = {
            ek: propdict({
                ak: (tf.identity(edges[ek][ak], name=('%s_copy'%ak)) if ak != 'layer' else np.copy(edges[ek][ak]))
                     for ak in edges[ek].keys()
            }) for ek in edges.keys()
        }

        graph_copy = Graph(nodes=nodes_copy, edges=edges_copy)
        return graph_copy

    def construct_graph(self,
                        nodes_list,
                        nodes_to_parents_list=None,
                        object_level_kwargs={},
                        **kwargs
    ):
        '''
        Builds the graph in the format used for dynamics models

        nodes_list: len(self.num_levels) <list> of tf.Tensors [B,T,N_level,D_level] <tf.float32>
        nodes_to_parents_list: len(self.num_levels - 1) <list> of tf.Tensors [B,T,N_level] <tf.int32> indices in range [0, N_(level+1) )
        '''
        self.num_levels = len(nodes_list) + int(self.build_object_nodes)

        if isinstance(nodes_list, OrderedDict):
            nodes_list = nodes_list.values()
        if isinstance(nodes_to_parents_list, OrderedDict):
            nodes_to_parents_list = nodes_to_parents_list.values()

        # compute edges up and down the hierarchy (effectively pointers to nodes)
        if nodes_to_parents_list is None:
            # TODO: find nearest neighbors
            raise NotImplementedError("This should just be a nearest neighbors function")

        # build object nodes
        if self.build_object_nodes:
            object_nodes, n2obj_inds = self.add_object_level(nodes_list[-1], **object_level_kwargs)
            nodes_list.append(object_nodes)
            nodes_to_parents_list.append(n2obj_inds)

        # convert the nodes to attrs
        all_nodes = {}
        attribute_dims_dict = {}
        attribute_preprocs_dict = {}
        for level, nodes in enumerate(nodes_list):
            node_key = 'nodes_level_' + str(level)
            if self.Dims is not None:
                Dims = self.Dims[level]
                node_attrs = Dims.get_tensor_from_attrs(nodes, Dims.keys(), postproc=True, concat=False,
                                                        stop_gradient={a:True for a in self.stop_gradient_on_attributes})
                node_attrs = propdict(node_attrs)
                new_attr_dims = copy.deepcopy({attr:Dims[attr][:2] for attr in Dims.keys()})
            else:
                node_attrs, new_attr_dims = self.vectors_to_attrs(nodes, self.attribute_dims[level], self.attribute_preprocs[level])
            all_nodes[node_key] = node_attrs
            # alter the attribute dims and preprocs
            attribute_dims_dict[node_key] = new_attr_dims
            attribute_preprocs_dict[node_key] = self.attribute_preprocs[level]
        self.attribute_dims = attribute_dims_dict
        self.attribute_preprocs = attribute_preprocs_dict

        all_edges = {}
        for level, n2p_inds in enumerate(nodes_to_parents_list):
            assert n2p_inds.shape.as_list() == nodes_list[level].shape.as_list()[:-1], (n2p_inds.shape.as_list(), nodes_list[level].shape.as_list())
            assert n2p_inds.dtype == tf.int32
            c2p_key = 'child_to_parent_edges_%d_to_%d' % (level, level+1)
            p2c_key = 'parent_to_child_edges_%d_to_%d' % (level+1, level)
            valid_nodes = tf.cast(all_nodes['nodes_level_'+str(level)]['valid'][...,0], tf.bool) # [B,T,N] in {False, True}

            # add edges to dict
            c2p_rect = graphical.add_batch_time_node_index(n2p_inds) # [B,T,N,4] tensor of (b_ind, t_ind, child_ind, parent_ind)
            c2p_list = tf.gather_nd(c2p_rect, tf.cast(tf.where(valid_nodes), tf.int32)) # [?, 4]
            all_edges[c2p_key] = propdict({
                'layer': np.array(['nodes_level_'+str(lev) for lev in [level, level+1]]).reshape([1,2]),
                'idx': c2p_list})

            p2c_list = tf.concat([c2p_list[:,0:2], c2p_list[:,3:4], c2p_list[:,2:3]], axis=-1)
            all_edges[p2c_key] = propdict({
                'layer': np.array(['nodes_level_'+str(lev) for lev in [level+1, level]]).reshape([1,2]),
                'idx': p2c_list})

        # now build graph
        self.graph = Graph(nodes=all_nodes, edges=all_edges)

class GraphModel(object):
    def __init__(self, **kwargs):
        self.model_params = kwargs
        self.G_pred = []
        self.graph_construction_kwargs = None

    def predict(self, nodes, edges, *args, **kwargs):
        raise NotImplementedError("Dynamics models must overwrite the predict method")

class PhysicsModel(GraphModel):

    def __init__(
            self,
            model_name=None,
            graph_construction_kwargs={},
            init_values={},
            graph_ops_list=[],
            history_kwargs=None,
            tracking_kwargs=None,
            **kwargs):


        # model scope
        self.model_name = model_name if model_name is not None else ''
        assert isinstance(self.model_name, str), "model_name must be a string to be added as a prefix to all scopes, but it's a %s" % type(self.model_name)

        # for building graph
        self.graph_construction_kwargs = copy.deepcopy(graph_construction_kwargs)

        # init values
        self.init_values = init_values

        # list of (func, kwargs)
        self.graph_ops_list = [(op.pop('func'), {k:op[k] for k in op.keys() if k != 'func'}) for op in copy.deepcopy(graph_ops_list)]

        # kwargs for the history-dependent op
        self.history_kwargs = history_kwargs

        # kwargs for tracking nodes over time
        self.tracking_kwargs = tracking_kwargs

        # op counter for scopes
        self.op_counter = 0

        # the predicted output graph
        self.G_pred = []

    def predict(self,
                input_nodes,
                input_edges=None,
                projection_matrix=None,
                num_rollout_times=1,
                output_node_keys=None,
                stop_all_gradients=False,
                dimension_dicts=None,
                num_levels = None,
                **kwargs):

        self.pmat = projection_matrix

        num_levels = num_levels or len(input_nodes)
        # if input_edges is not None:
        #     assert len(input_edges) == num_levels - 1

        if stop_all_gradients:
            print("stopping gradients")
            input_nodes = [tf.stop_gradient(nodes) for nodes in input_nodes]
            input_edges = [tf.stop_gradient(edges) for edges in input_edges]

        # build the graph
        with tf.variable_scope(self.model_name + "GraphConstruction", reuse=True):

            self.G = HierarchicalVisualGraphConstructor(
                num_levels=num_levels, dimension_dicts=dimension_dicts,
                **self.graph_construction_kwargs)
            self.G.construct_graph(input_nodes, input_edges, **self.graph_construction_kwargs)

        # print("Constructed Graph")
        # print(self.G.graph)

        # update state with recurrence and prediction of invisible nodes
        with tf.variable_scope(self.model_name + "GraphStateInference", reuse=True):
            if self.history_kwargs is not None:
                self.G.graph = self.infer_state(self.G.graph, **self.history_kwargs)
            # compute tracklets and add delta_positions/matching costs
            if self.tracking_kwargs is not None:
                self.G.graph = self.infer_tracking(self.G.graph, **self.tracking_kwargs)

        # dynamics
        with tf.variable_scope(self.model_name + "GraphDynamics", reuse=True):
            # initialize the output
            self.G_pred = []
            if self.pmat is not None:
                self.pmat = self.pmat[:,-1:]
            G_init = self.G.copy_graph(self.get_final_graph_state(self.G.graph))
            G_init = self.init_graph(G_init)

            # rollout
            for t in range(num_rollout_times):
                G_next = self.compute_dynamics(G_init)
                self.G_pred.append(G_next)
                G_init = self.G_pred[-1]

        output_vectors = self.get_output_vectors(self.G_pred, output_node_keys)
        return output_vectors

    def infer_state(self, graph, update_levels=[-1], **kwargs):

        # compute new effects at each level
        for level in update_levels:
            node_key = 'nodes_level_' + str(level % self.G.num_levels)
            nodes = graph['nodes'].pop(node_key)
            vectors = nodes['vector']
            with tf.variable_scope("history_update_"+node_key, reuse=True):
                vectors = IntegratedGraphCell.recurrent_graphconv(vectors, **kwargs)
                graph['nodes'][node_key] = self.G.vectors_to_attrs(vectors, self.G.attribute_dims[node_key], self.G.attribute_preprocs[node_key])[0]

        return graph

    def infer_tracking(self, graph, tracking_level=-1, match_dist_thresh=None, thresh_dims_list=[[0,3]], delta_dims={'positions': [0,3]}, delta_funcs={'positions': tf.subtract}, **kwargs):

        node_key = 'nodes_level_' + str(tracking_level % self.G.num_levels)
        nodes = graph.nodes[node_key]
        B,T,N,D = nodes['vector'].shape.as_list()
        new_attrs = {'matches_valid': [tf.zeros([B,N,1], tf.float32)], 'cost': [tf.zeros([B,N,1], tf.float32)]}
        for attr,dims in delta_dims.items():
            new_attrs['delta_'+attr] = [tf.zeros([B,N,(dims[1]%D)-(dims[0]%D)], tf.float32)]

        # match up pairs of nodes
        for t in range(T-1):
            nodes_prev = nodes['vector'][:,t]
            nodes_now = nodes['vector'][:,t+1]
            match = graphical.hungarian_node_matching(nodes_now, nodes_prev, **kwargs) # [B,N] indices into nodes_prev
            nodes_prev = graphical.permute_nodes(nodes_prev, match)

            # compute new attributes from matching
            matches_valid = nodes_now[...,-1:] * nodes_prev[...,-1:] # [B,N,1]

            if match_dist_thresh is not None:
                dists2 = tf.square(
                    tf.concat([nodes_prev[...,td[0]:td[1]] for td in thresh_dims_list], axis=-1) -
                    tf.concat([nodes_now[...,td[0]:td[1]] for td in thresh_dims_list], axis=-1))
                dists2 = tf.reduce_sum(dists2, axis=-1, keepdims=True)
                matches_valid = matches_valid * tf.cast(dists2 < match_dist_thresh, tf.float32)
            new_attrs['matches_valid'].append(matches_valid)
            matches_cost = graphical.matching_cost(nodes_prev, nodes_now, **kwargs) # [B,N,1]
            new_attrs['cost'].append(matches_cost)

            for attr,dims in delta_dims.items():
                d_func = delta_funcs.get(attr, tf.subtract)
                d_attr = d_func(nodes_now[...,dims[0]:dims[1]], nodes_prev[...,dims[0]:dims[1]])
                new_attrs['delta_'+attr].append(d_attr)

        # concat all new attrs and add to graph
        for k in new_attrs.keys():
            d_attr = tf.stop_gradient(tf.stack(new_attrs[k], axis=1)) # [B,T,N,delta_dims]
            graph.nodes[node_key][k] = d_attr

        return graph

    def init_graph(self, graph):

        for node_key in self.init_values.keys():
            for attr, value in self.init_values[node_key].items():
                init_value = tf.constant(value, tf.float32) * tf.ones_like(graph.nodes[node_key][attr])
                graph = self.update_vector_and_attr(graph, init_value, node_key, attr)
                print("initialized %s/%s" % (node_key, attr))

        return graph

    def compute_dynamics(self, graph, **kwargs):

        # copy the graph
        graph = self.G.copy_graph(graph)

        # graph update
        print("List of Graph Computations in Dynamics Model")
        print("============================================")
        for (func, kwargs) in self.graph_ops_list:
            op_type, kwargs_here = self.parse_op_kwargs(graph, kwargs)
            with tf.variable_scope(self.op_scope(), reuse=True):
                print(tf.get_variable_scope().name, func.__name__)
                op_here = op_type(func) # wrapper func w/ signature op(graph, edge_key, sender_feat_keys, receiver_feat_keys)
                graph = op_here(graph, **kwargs_here) # adds new effects to graph

        # reset the op counter for the next rollout time
        self.op_counter = 0

        return graph

    def op_scope(self):
        scope = 'dynamics_op_' + str(self.op_counter)
        self.op_counter += 1
        return scope

    def parse_op_kwargs(self, graph, kwargs):
        kwargs_here = copy.deepcopy(kwargs)
        edge_key = None
        op_type = kwargs_here.pop('op_type', 'graph')
        if op_type == 'graph':
            op_type = graph_op
            edge_key = kwargs_here.get('edge_key', None)
            assert edge_key is not None
        elif op_type == 'node':
            op_type = node_op
        elif op_type == 'edge':
            op_type = edge_op
            edge_key = kwargs_here.get('edge_key', None)
            assert edge_key is not None
        elif op_type == 'transform':
            op_type = self.transform_attr_by_effects
        else:
            raise ValueError("op_type must be in ['graph', 'node', 'edge', 'transform'] but is %s" % op_type)

        # construct edges on the fly
        if edge_key is not None:
            if 'all_to_all' in edge_key and edge_key not in graph.edges.keys():
                graph = self.G.all_to_all_edges_from_nodes(graph, edge_key)
            elif 'across_parent' in edge_key and edge_key not in graph.edges.keys():
                graph = self.G.across_parent_edges_from_nodes(graph, edge_key)
            elif 'within_parent' in edge_key and edge_key not in graph.edges.keys():
                graph = self.G.within_parent_edges_from_nodes(graph, edge_key)

        # add pmat
        if kwargs_here.get('use_pmat', False):
            kwargs_here['pmat'] = self.pmat

        return op_type, kwargs_here

    def get_vector_dims(self, graph, node_key, attr_key):
        assert isinstance(self.G, HierarchicalVisualGraphConstructor)
        attr_dims = self.G.attribute_dims[node_key][attr_key]
        return graph.nodes[node_key]['vector'][...,attr_dims[0]:attr_dims[1]]

    def update_vector_and_attr(self, graph, new_values, node_key, attr_key):
        assert isinstance(self.G, HierarchicalVisualGraphConstructor)
        attr_dims = self.G.attribute_dims[node_key][attr_key]
        attr_preproc = self.G.attribute_preprocs[node_key].get(attr_key, tf.identity) or tf.identity

        vectors = graph['nodes'][node_key].pop('vector')
        assert new_values.shape.as_list()[-1] == (attr_dims[1] - attr_dims[0]), "new values must have same dims as old ones"

        # first update the vectors in place
        graph['nodes'][node_key]['vector'] = tf.concat([
            vectors[...,:attr_dims[0]],
            new_values,
            vectors[...,attr_dims[1]:]], axis=-1)

        # now update the attrs
        graph['nodes'][node_key].pop(attr_key)
        graph['nodes'][node_key][attr_key] = attr_preproc(new_values)

        return graph

    def transform_attr_by_effects(self, func=None):
        '''
        takes effects and applies to vector and its associated attr using func(effects, attr)
        '''
        def wrapper(graph, node_key, attr_key, effects_key='effects', reduce_dim=False, *args, **kwargs):
            effects = graph['nodes'][node_key][effects_key]
            E = effects.shape.as_list()[-1] # effects dim

            attr_keys = [attr_key] if not isinstance(attr_key, list) else attr_key
            for attr_key in attr_keys:
                old_attr = graph['nodes'][node_key].get(attr_key, None)
                D = old_attr.shape.as_list()[-1] if old_attr is not None else None

                if (D is not None and D != E and E != 1 and func.__name__ == 'add') or reduce_dim: # linear layer
                    effects_here = mlp(effects, num_features=[D], activations=None, scope=('linear_dimension_match_%s' % attr_key))
                else:
                    effects_here = effects

                # print(attr_key, effects_key, effects_here.shape.as_list(), tf.get_variable_scope().name)

                # update an existing attr and its support in the input vector
                if attr_key in self.G.attribute_dims[node_key].keys():
                    old_vals = self.get_vector_dims(graph, node_key, attr_key)
                    new_vals = func(effects_here, old_vals, *args, **kwargs)
                    if attr_key == 'velocities' and 'effects' in effects_key and PRINT:
                        new_vals = tf.Print(new_vals, [tf.reduce_max(tf.abs(new_vals)), tf.reduce_max(tf.abs(old_vals))], message='new_vs/old_vs/%s' % effects_key)
                    graph = self.update_vector_and_attr(graph, new_vals, node_key, attr_key)
                    # print("updated attr", node_key, attr_key)
                else: # add a new attr
                    old_vals = graph.nodes[node_key].get(attr_key, tf.zeros_like(effects))
                    new_vals = func(effects_here, old_vals, *args, **kwargs)
                    graph.nodes[node_key][attr_key] = new_vals
            return graph
        return wrapper

    @staticmethod
    def scatter_effects(
            edge_inds, sender_attrs, receiver_attrs, edge_features, agg_method='sum', **kwargs):
        '''
        Use scatter_nd to supply receivers with the effects from senders, summing contributions from each edge

        edge_inds: [?,4] list of edges (b_ind, t_ind, sender_ind, receiver_ind)
        '''
        assert agg_method in ['sum', 'mean', 'max', 'min']
        sender_vectors = tf.concat([sender_attrs[k] for k in sender_attrs.keys()], axis=-1)

        out_shape = receiver_attrs[receiver_attrs.keys()[0]].shape.as_list()[:-1] + sender_vectors.shape.as_list()[-1:]

        if agg_method == 'sum':
            receiver_effects = tf.scatter_nd(
                indices=tf.concat([edge_inds[:,0:2], edge_inds[:,3:4]], -1),
                updates=tf.gather_nd(sender_vectors, edge_inds[:,0:3]),
                shape=out_shape)
        else:
            raise NotImplementedError("Do stuff with unsorted_segment_op")

        return receiver_effects, None

    @staticmethod
    def project_velocity_hw(node_attrs, pmat, near_plane=-0.2, **kwargs):
        xyz = node_attrs['positions']
        vels = node_attrs['velocities']
        assert xyz.shape.as_list()[-1] == vels.shape.as_list()[-1] == 3, (xyz.shape.as_list(), vels.shape.as_list())
        assert pmat.shape.as_list() == xyz.shape.as_list()[:2] + [4,4], (pmat.shape.as_list(), xyz.shape.as_list())

        hw_start = agent_particles_to_image_coordinates(
            xyz, Pmat=pmat, H_out=1, W_out=1, to_integers=False)
        hw_start = (hw_start * 2.) - 1.

        xyz_end = xyz + vels
        xyz_end = tf.concat([xyz_end[...,0:2], tf.minimum(xyz_end[...,2:3], near_plane)], axis=-1)
        hw_end = agent_particles_to_image_coordinates(
            xyz_end, Pmat=pmat, H_out=1, W_out=1, to_integers=False)
        hw_end = (hw_end * 2.) - 1.
        hw_vels = hw_end - hw_start

        return hw_vels

    @staticmethod
    def graphconv_from_attrs(
            edge_inds, sender_attrs, receiver_attrs, edge_features, diff_attrs=[], diff_funcs={}, **kwargs):
        '''
        edge_inds: [?,4] list of edges (b_ind, t_ind, sender_ind, receiver_ind)
        '''
        for attr in diff_attrs:
            assert attr in sender_attrs.keys() and attr in receiver_attrs.keys()
            diff_func = diff_funcs.get(attr, tf.subtract)
            diff = diff_func(sender_attrs[attr], receiver_attrs[attr])
            sender_attrs['diff_'+attr] = diff

        sender_vectors = tf.concat([sender_attrs[k] for k in sender_attrs.keys() if k not in diff_attrs], axis=-1)
        receiver_vectors = tf.concat([receiver_attrs[k] for k in receiver_attrs.keys() if k not in diff_attrs], axis=-1)
        # print("senders, receivers, diff_attrs", sender_vectors.shape.as_list(), receiver_vectors.shape.as_list(), diff_attrs)

        receiver_effects, edge_effects = graphical.graphconv_pairwise_from_inds(
            lnodes=sender_vectors, # e.g. children
            rnodes=receiver_vectors, # e.g. parents
            edges=edge_inds, # e.g. c2p_edges
            right_receivers=True,
            edge_effects=True,
            scope='gconv_pairwise',
            **kwargs)

        return receiver_effects, edge_effects

    @staticmethod
    def features_from_attrs(inp_attrs, node_dims, activation=None, **kwargs):
        '''
        simple wrapper to get new features via mlp from some attrs
        '''
        inp = tf.concat([inp_attrs[k] for k in inp_attrs.keys()], axis=-1)
        feats = mlp(inp, num_features=node_dims, activation=activation, scope='feats_from_attrs_mlp', **kwargs)
        return feats

    @staticmethod
    def interacting_edges_by_distance(edge_inds, senders, receivers, edge_features, valid_key='valid', distance_thresh=1.0, binarize=True, eps=1e-6, **kwargs):

        assert any([k in senders.keys() for k in ['positions', 'hw_centroids']])
        assert any([k in receivers.keys() for k in ['positions', 'hw_centroids']])
        sender_vectors = tf.concat([senders[k] for k in senders.keys() if k != valid_key], axis=-1)
        receiver_vectors = tf.concat([receivers[k] for k in receivers.keys() if k != valid_key], axis=-1)
        sender_valid = senders.get(valid_key, tf.ones_like(sender_vectors[...,-1:]))
        receiver_valid = receivers.get(valid_key, tf.ones_like(receiver_vectors[...,-1:]))

        sender_positions = tf.gather_nd(sender_vectors, edge_inds[:,0:3]) # [?, 3]
        receiver_positions = tf.gather_nd(receiver_vectors, tf.concat([edge_inds[:,0:2], edge_inds[:,3:4]], axis=-1)) # [?, 3]
        sender_valid = tf.gather_nd(sender_valid, edge_inds[:,0:3])
        receiver_valid = tf.gather_nd(receiver_valid, tf.concat([edge_inds[:,0:2], edge_inds[:,3:4]], axis=-1)) # [?, 3]
        valid = tf.logical_and(sender_valid > 0.5, receiver_valid > 0.5)

        dists = tf.sqrt(tf.reduce_sum(tf.square(sender_positions - receiver_positions), axis=-1, keepdims=True) + eps)
        if binarize:
            interacting = tf.logical_and(dists < distance_thresh, valid) # [?, 1] <tf.bool>
        else:
            interacting = 1.0 / (1.0 + (dists/distance_thresh)) # [?, 1] <tf.float> in [0., 1.]
            interacting = interacting * tf.cast(valid, tf.float32)

        return None, interacting

    @staticmethod
    def collide_interacting_nodes(edge_inds, senders, receivers, interacting_edges, **kwargs):
        interacting_edges = interacting_edges[0]
        assert interacting_edges.shape.as_list()[-1] == 1
        if interacting_edges.dtype == tf.float32:
            interacting_edges = interacting_edges > 0.5
        assert interacting_edges.dtype == tf.bool
        interacting_edge_inds = tf.cast(tf.where(interacting_edges[:,0]), tf.int32) # [num_inter_edges, 1]
        interacting_edge_inds = tf.gather_nd(edge_inds, interacting_edge_inds) # [num_inter_edges, 4]

        if PRINT:
            interacting_edge_inds = tf.Print(
                interacting_edge_inds,
                [tf.shape(interacting_edge_inds)[0], tf.shape(edge_inds)[0], tf.square(tf.shape(senders[senders.keys()[0]])[2])],
                message='colliding, across, nodes**2')

        receiver_effects, _ = PhysicsModel.graphconv_from_attrs(
            interacting_edge_inds, senders, receivers, None, **kwargs)

        return receiver_effects, None

    @staticmethod
    def get_final_graph_state(G):
        '''
        Build a graph that only uses the final time step of the node vectors and the edges
        '''
        all_nodes = G['nodes']
        last_nodes = {}
        for nk, nodes in all_nodes.items():
            node_attrs = {}
            for attr, vals in nodes.items():
                node_attrs[attr] = vals[:,-1:]
                num_times = vals.shape.as_list()[1]
            last_nodes[nk] = propdict(node_attrs)

        all_edges = G['edges']
        final_edges = {}
        for ek, edges in all_edges.items():
            edge_attrs = {}
            last_time_inds = tf.cast(tf.where(tf.equal(edges['idx'][:,1], tf.ones_like(edges['idx'][:,1]) * (num_times-1))), tf.int32) # [num_last_edges,1]
            for attr, vals in edges.items():
                if attr == 'layer':
                    edge_attrs[attr] = vals
                elif attr == 'idx':
                    last_edges = tf.gather_nd(vals, last_time_inds) # [num_last_edges, 4]
                    last_edges = tf.concat([last_edges[:,0:1], tf.zeros_like(last_edges[:,0:1]), last_edges[:,2:]], axis=-1) # all time inds are 0
                    edge_attrs[attr] = last_edges
                else:
                    edge_attrs[attr] = tf.gather_nd(vals, last_time_inds)
            final_edges[ek] = propdict(edge_attrs)

        return Graph(nodes=last_nodes, edges=final_edges)

    def get_output_vectors(self, graphs_list, output_node_keys=None):
        '''
        Gets the 'vector' attributes of each graph at each time and concats
        '''
        if output_node_keys is None:
            output_node_keys = graphs_list[0].nodes.keys()

        outputs = {k: [] for k in output_node_keys}

        # list of dicts
        for t in range(len(graphs_list)):
            for k in output_node_keys:
                outputs[k].append(graphs_list[t].nodes[k]['vector'])

        # concat
        outputs = OrderedDict([
            (k, tf.concat(outputs[k], axis=1))
            for k in sorted(outputs.keys())])

        return outputs

class VisualPhysicsModel(object):

    def __init__(
            self,
            encoder_model_func,
            encoder_model_params,
            physics_model_class=None,
            physics_model_params={},
            **kwargs
    ):
        '''
        Encodes a movie-like input with encoder_model_func(inputs, **encoder_model_params)
        Takes their output and constructs a Graph intermediate
        Passes this Graph to a learnable PhysicsModel for forward prediction 
        '''
        self.encoder = encoder_model_func
        self.encoder_params = encoder_model_params
        self.nodes = None
        self.edges = None
        self.losses = {}
        self.Dims = None

        if physics_model_class is not None:
            self.physics_model = physics_model_class(**physics_model_params)
        else:
            self.physics_model = GraphModel(**physics_model_params)

    def encode(self, inputs, train_targets, input_times=None, train=True):

        if input_times is not None:
            visible_inputs = {k:inputs[k][:,:input_times] for k in inputs}
        else:
            visible_inputs = inputs

        with tf.variable_scope('VisualEncoder'):
            outputs, enc_params = self.encoder(
                inputs=visible_inputs,
                train_targets=train_targets,
                train=train,
                **self.encoder_params
            )

        # parse outputs

        nodes = {k:outputs.pop(k) for k in outputs.keys() if 'nodes' in k}
        self.nodes = OrderedDict(sorted(nodes.items())) # assumes nodes are keyed hierarchically, _0, _1, etc.
        self.node_ndims = [n.shape.as_list()[-1] for k,n in self.nodes.items()]
        edges = {k:outputs.pop(k) for k in outputs.keys() if 'segments' in k or 'edges' in k}
        self.edges = OrderedDict(sorted(edges.items())) # assumes edges are keyed hierarchically
        Dims = {k:outputs.pop(k) for k in outputs.keys() if 'dimensions' in k}
        self.Dims = OrderedDict(sorted(Dims.items()))
        self.losses.update({k:outputs.pop(k) for k in outputs.keys() if 'loss' in k})

        # print("nodes", self.nodes)
        # print("node_ndims", self.node_ndims)
        # print("edges", self.edges)
        # print("Dims", self.Dims)
        # print("Losses", self.losses)
        # print("Remaining Outputs", outputs)

        return self.nodes, self.edges

    def decode_attributes(self, nodes, segments, labels, dimension_dict,
                          losses_dict=OrderedDict()
    ):

        '''
        Compute all losses and predicted images that depend on the graph encoding

        nodes: [B,T,N,D] the nodes from a single level of the hierarchy
        segments: [B,T,H,W] the segments that indicate which hw positions correspond to which node
        dimension_dict: a <DimensionDict> that indicates which node dimensions correspond to which attributes and how to postprocess them
        labels: a dict of ground truth tensors to use for each loss
        losses: an OrderedDict of dicts: losses_dict[loss_name] = dict({'func', logits_keys <list>, labels_keys <list>, loss_func_kwargs})
        '''
        losses = {}
        decoded_images = {}
        def _id_decoder(attrs, segments, **kwargs):
            return attrs

        for loss_name, loss_dict in losses_dict.items():
            attrs = loss_dict['logits_keys'] # list
            targets = loss_dict['labels_keys'] # list
            decoder = loss_dict.get('decoder_func', _id_decoder)
            decoder_kwargs = loss_dict.get('decoder_kwargs', {})
            loss_func = loss_dict['loss_func'] # func
            loss_kwargs = loss_dict.get('loss_func_kwargs', {})

            # get attr logits using dimension_dict
            attr_dict = dimension_dims.get_tensor_from_attrs(nodes, attrs, postproc=True, concat=False)

            # pass through the decoder to return a dict of decodes
            decoded_attrs = decoder(attr_dict, segments, **decoder_kwargs)

            # get labels

        return losses, decoded_images

    def __call__(self, inputs, train_targets, train=True, visible_times=None, num_levels=None,
                 dynamics_targets=['projection_matrix'], prediction_kwargs={},
                 dynamics_loss_func=None, dynamics_loss_func_kwargs={},
                 **kwargs):

        # encode the movie
        nodes, edges = self.encode(inputs, train_targets, input_times=visible_times, train=train)

        # update inputs to graph
        self.B, self.T_vis = nodes[nodes.keys()[0]].shape.as_list()[:2]
        prediction_kwargs['dimension_dicts'] = self.Dims
        prediction_kwargs['num_levels'] = num_levels
        tstart = self.T_vis - prediction_kwargs.get('max_rollout_times', 1)

        predictions = []
        if self.physics_model is not None:
            vis_nodes = OrderedDict([
                (k, Ns[:,:tstart]) for k,Ns in nodes.items()])
            vis_n2p_inds = OrderedDict([
                (k, tf.reshape(segs, [self.B,self.T_vis,-1])[:,:tstart])
                for k,segs in edges.items() if 'segments' in k])
            dynamics_inputs = [inputs[k][:,:tstart] for k in dynamics_targets]
            predictions = self.physics_model.predict(
                vis_nodes, vis_n2p_inds, *dynamics_inputs,
                num_rollout_times=1,
                **prediction_kwargs)

        # print("predictions")
        # print(predictions)

        # use predictions to generate a loss
        if dynamics_loss_func is not None and len(predictions):
            pred_nodes = predictions[predictions.keys()[-1]]
            gt_nodes = nodes[nodes.keys()[-1]][:,-1:]
            Dims = self.Dims[self.Dims.keys()[-1]]

            # print("pred_nodes", pred_nodes)
            # print("gt_nodes", gt_nodes)
            # print("Dims", Dims)
            dyn_loss = dynamics_loss_func(logits=pred_nodes, labels=gt_nodes,
                                          dimension_dict=Dims,
                                          **dynamics_loss_func_kwargs)
            # print("dyn loss", dyn_loss)
            # reduce across batch
            self.losses['dynamics_loss'] = {'dynamics_loss': tf.reduce_mean(dyn_loss)}
        else:
            self.losses['dynamics_loss'] = {'dynamics_loss': tf.reduce_mean(nodes[nodes.keys()[-1]])}

        if train:
            outputs = self.losses
        else:
            outputs = {'dynamics_loss': dyn_loss}
            outputs.update({k:self.nodes[k] for k in self.nodes})
            outputs.update({k:self.edges[k] for k in self.edges})
            outputs.update({k:self.Dims[k] for k in self.Dims})
            outputs.update({k+'_pred':p for k,p in predictions.items()})

        return outputs, {}
