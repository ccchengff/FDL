#-*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf
from fdl.tensorflow.comm_ops import send_op, recv_op
import logging

class FLHostGraph(object):

    def __init__(self, align_keys=None):
        super(FLHostGraph, self).__init__()
        self._send_bottoms = {}
        self._align_keys = align_keys
        self._check_align_op = self._check_align_keys()
    
    def _check_align_keys(self):
        if self._align_keys is not None:
            hash_key = tf.strings.to_hash_bucket_fast(self._align_keys, 2 ** 31 - 1)
            guest_hash_keys = recv_op("HashedAlignKeys", src_party="guest", dtype=hash_key.dtype)
            return tf.assert_equal(guest_hash_keys, hash_key, name="CheckAlignKeys")
        else:
            return tf.no_op(name="CheckAlignKeys")
    
    def send_bottom(self, name, tensor, dst_parties="guest", requires_grad=True):
        if name in self._send_bottoms:
            raise RuntimeError("Name %s already exists" % name)
        if isinstance(dst_parties, (list, tuple)):
            raise RuntimeError("Currently we only support sending to one party")
        op = send_op(name, tensor, dst_parties=dst_parties)
        self._send_bottoms[name] = (tensor, dst_parties, requires_grad, op)
    
    def minimize(self, 
                 optimizer, 
                 global_step=None, 
                 gate_gradients=tf.train.Optimizer.GATE_OP,
                 aggregation_method=None,
                 colocate_gradients_with_ops=False,
                 name=None, 
                 return_grads=False):
        # send forward activations to guest, 
        # receive backward derivatives from guest, 
        # and compute gradients for bottom models
        if len(self._send_bottoms) == 0:
            raise ValueError("No activations to be sent")
        send_ops = []
        grads_and_vars = [] # gradients and variables
        devs_and_acts = [] # derivatives and activations for the outputs of bottoms
        for name, (tensor, dst_party, requires_grad, op) in self._send_bottoms.items():
            send_ops.append(op)
            if requires_grad:
                with tf.control_dependencies([op]):
                    tensor_grad = recv_op(name + "Grad", src_party=dst_party, dtype=tensor.dtype)
                    g_a_v = optimizer.compute_gradients(
                        tensor, 
                        gate_gradients=gate_gradients,
                        aggregation_method=aggregation_method,
                        colocate_gradients_with_ops=colocate_gradients_with_ops, 
                        grad_loss=tensor_grad)
                    devs_and_acts.append((tensor_grad, tensor))
                    grads_and_vars.extend(g_a_v)
        grads_and_vars = [gv for gv in grads_and_vars if gv[0] is not None]
        
        # if there are two or more bottom models, 
        # we should coalesce gradients for the same tensor
        named_vars = list({x[1].name : x[1] for x in grads_and_vars}.items())
        coalesce_grads = [[x[0] for x in grads_and_vars if x[1].name == name] \
                          for name, _ in named_vars]
        coalesce_grads = [x[0] if len(x) == 1 else tf.add_n(x) for x in coalesce_grads]
        grads_and_vars = list(zip(coalesce_grads, map(lambda x: x[1], named_vars)))

        if len(grads_and_vars) > 0:
            train_op = optimizer.apply_gradients(
                grads_and_vars, 
                global_step=global_step)
        else:
            train_op = tf.no_op()
        train_op = tf.group(
            [self._check_align_op, train_op, *send_ops], 
            name=name)
        
        if return_grads:
            return train_op, grads_and_vars, devs_and_acts
        else:
            return train_op
    
    def predict(self, name=None):
        # send forward activations to guest
        if len(self._send_bottoms) == 0:
            raise ValueError("No activations to be sent")
        send_ops = [op for _, _, _, op in self._send_bottoms.values()]
        pred_op = tf.group([self._check_align_op] + send_ops, name=name)
        return pred_op
    
    def local_minimize(self, 
                       optimizer, 
                       devs, 
                       cache_acts, 
                       sim_thres, 
                       global_step=None, 
                       gate_gradients=tf.train.Optimizer.GATE_OP,
                       aggregation_method=None,
                       colocate_gradients_with_ops=False,
                       name=None):
        # compute weighted derivative according to similarities
        if len(self._send_bottoms) == 0:
            raise ValueError("No activations to be sent")
        weighted_devs = {}
        if sim_thres is not None and (-1.0 < sim_thres < 1.0):
            logging.info(f"Weight instances for sim_thres = {sim_thres}")
            acts, cache = [], []
            for name, (tensor, dst_party, requires_grad, op) in self._send_bottoms.items():
                if requires_grad:
                    if name not in cache_acts:
                        raise ValueError("No cache provided for " + name)
                    acts.append(tensor)
                    cache.append(cache_acts[name])
            if len(acts) == 1:
                acts = acts[0]
                cache = cache[0]
            else:
                acts = tf.concat(acts, axis=1)
                cache = tf.concat(cache, axis=1)
            similarities = _compute_cosine_similarity(acts, cache)
            ins_weights = similarities * tf.cast(similarities >= sim_thres, tf.float32)
            for name, dev in devs.items():
                weighted_devs[name] = tf.transpose(tf.multiply(
                    tf.transpose(devs[name]), ins_weights))
        else:
            logging.info(f"Will not weight instances for sim_thres = {sim_thres}")
            ins_weights = None
            weighted_devs = devs
        
        grads_and_vars = []
        for name, weighted_dev in weighted_devs.items():
            tensor = self._send_bottoms[name][0]
            g_a_v = optimizer.compute_gradients(
                tensor, 
                gate_gradients=gate_gradients,
                aggregation_method=aggregation_method,
                colocate_gradients_with_ops=colocate_gradients_with_ops, 
                grad_loss=weighted_dev)
            grads_and_vars.extend(g_a_v)
        
        grads_and_vars = [gv for gv in grads_and_vars if gv[0] is not None]

        # if there are two or more bottom models, 
        # we should coalesce gradients for the same tensor
        named_vars = list({x[1].name : x[1] for x in grads_and_vars}.items())
        coalesce_grads = [[x[0] for x in grads_and_vars if x[1].name == name] \
                          for name, _ in named_vars]
        coalesce_grads = [x[0] if len(x) == 1 else tf.add_n(x) for x in coalesce_grads]
        grads_and_vars = list(zip(coalesce_grads, map(lambda x: x[1], named_vars)))

        if len(grads_and_vars) > 0:
            train_op = optimizer.apply_gradients(
                grads_and_vars, 
                global_step=global_step, 
                name=name)
        else:
            train_op = tf.no_op(name=name)
        
        return train_op, ins_weights


class FLGuestGraph(object):

    def __init__(self, align_keys=None, deps=[]):
        super(FLGuestGraph, self).__init__()
        self._align_keys = align_keys
        self._dependencies = deps
        if self._align_keys is not None:
            self._dependencies.append(self._align_keys)
        if len(self._dependencies) == 0:
            raise ValueError("Please provide align_keys or deps as dependencies")
        self._check_align_op = self._check_align_keys()
        self._remote_bottoms = {}
    
    def _check_align_keys(self):
        if self._align_keys is not None:
            hash_key = tf.strings.to_hash_bucket_fast(self._align_keys, 2 ** 31 - 1)
            return send_op("HashedAlignKeys", hash_key, dst_parties="auto", op_name="CheckAlignKeys")
        else:
            return tf.no_op(name="CheckAlignKeys")
    
    def remote_bottom(self, name, src_party="host", dtype=tf.float32, shape=None, 
                      requires_grad=True):
        with tf.control_dependencies(self._dependencies):
            tensor = recv_op(name, src_party=src_party, dtype=dtype, shape=shape)
        # TODO: prefix the name by the current scope?
        self._remote_bottoms[name] = (tensor, src_party, requires_grad)
        return tensor

    def minimize(self, 
                 optimizer, 
                 loss, 
                 gate_gradients=tf.train.Optimizer.GATE_OP, 
                 aggregation_method=None, 
                 colocate_gradients_with_ops=False, 
                 grad_loss=None, 
                 global_step=None, 
                 name=None, 
                 perturb_fn=None, 
                 return_grads=None):
        # compute gradients
        remote_vars = [tensor for tensor, _, requires_grad \
                       in self._remote_bottoms.values() \
                       if requires_grad]
        var_list = remote_vars + \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) + \
            tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)
        all_grads_and_vars = optimizer.compute_gradients(
            loss, 
            var_list=var_list, 
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops, 
            grad_loss=grad_loss)
        all_grads_and_vars = [gv for gv in all_grads_and_vars if gv[0] is not None]
        if len(all_grads_and_vars) == 0:
            raise ValueError("No gradients provided for any variable")
        
        # send backward derivatives to other parties 
        # and apply gradients to local party
        remote_names = set(self._remote_bottoms.keys())
        send_ops = []
        grads_and_vars = [] # gradients and variables
        devs_and_acts = [] # derivatives and activations for the outputs of host bottoms
        for grad, var in all_grads_and_vars:
            tensor_name = var.name.split(".RecvFrom.", 2)[0]
            if tensor_name in remote_names:
                tensor, src_party, requires_grad = self._remote_bottoms[tensor_name]
                if var is not tensor:
                    raise RuntimeError(f"Got different vars: {var} vs. {tensor}")
                if requires_grad:
                    if perturb_fn is not None:
                        perturbed_grad = perturb_fn(grad)
                        send_ops.append(send_op(tensor_name + "Grad", perturbed_grad, src_party))
                    else:
                        send_ops.append(send_op(tensor_name + "Grad", grad, src_party))
                devs_and_acts.append((grad, var))
            else:
                grads_and_vars.append((grad, var))
        if len(grads_and_vars) > 0:
            train_op = optimizer.apply_gradients(
                grads_and_vars, 
                global_step=global_step)
        else:
            train_op = tf.no_op()
        train_op = tf.group(
            [self._check_align_op, train_op, *send_ops], 
            name=name)
        
        if return_grads:
            return train_op, grads_and_vars, devs_and_acts
        else:
            return train_op
        
    def predict(self, predictions, name=None):
        with tf.control_dependencies([self._check_align_op]):
            return tf.identity(predictions, name=name)
    
    def local_minimize(self, 
                       optimizer, 
                       ins_loss, 
                       acts, 
                       cache_devs, 
                       sim_thres, 
                       gate_gradients=tf.train.Optimizer.GATE_OP, 
                       aggregation_method=None, 
                       colocate_gradients_with_ops=False, 
                       global_step=None, 
                       name=None):
        avg_loss = tf.reduce_mean(ins_loss)
        
        # compute weighted loss according to similarities
        if sim_thres is not None and (-1.0 < sim_thres < 1.0):
            logging.info(f"Weight instances for sim_thres = {sim_thres}")
            # compute gradients for remote activations only
            remote_names_and_vars = []
            for name, act in acts.items():
                requires_grad = self._remote_bottoms[name][2]
                if requires_grad:
                    remote_names_and_vars.append((name, act))
            
            remote_grads_and_vars = optimizer.compute_gradients(
                avg_loss, 
                var_list=[nv[1] for nv in remote_names_and_vars], 
                gate_gradients=gate_gradients,
                aggregation_method=aggregation_method,
                colocate_gradients_with_ops=colocate_gradients_with_ops)
                        
            if len(remote_grads_and_vars) == 0:
                raise ValueError("No gradients provided for any remote activations")
            elif len(remote_grads_and_vars) == 1:
                grads = remote_grads_and_vars[0][0]
                cache = cache_devs[remote_names_and_vars[0][0]]
            else:
                cache = tf.concat([cache_devs[nv[0]] for nv in remote_names_and_vars], axis=1)
                grads = tf.concat([gv[0] for gv in remote_grads_and_vars], axis=1)
            similarities = _compute_cosine_similarity(grads, cache)
            ins_weights = similarities * tf.cast(similarities >= sim_thres, tf.float32)
            weighted_loss = tf.reduce_mean(ins_loss * ins_weights)
        else:
            logging.info(f"Will not weight instances for sim_thres = {sim_thres}")
            ins_weights = None
            weighted_loss = avg_loss
        
        # update local variables
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) + \
            tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)
        train_op = optimizer.minimize(
            weighted_loss, 
            var_list=var_list, 
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops, 
            grad_loss=None, 
            global_step=global_step, 
            name=name)

        return train_op, ins_weights


def _compute_cosine_similarity(x, y, epsilon=1e-12):
    dot = tf.reduce_sum(tf.multiply(x, y), axis=1)
    x_norm = tf.math.maximum(tf.math.sqrt(tf.reduce_sum(tf.math.pow(x, 2), axis=1)), epsilon)
    y_norm = tf.math.maximum(tf.math.sqrt(tf.reduce_sum(tf.math.pow(y, 2), axis=1)), epsilon)
    ret = dot / tf.multiply(x_norm, y_norm)
    ret = tf.clip_by_value(ret, -1.0, 1.0)
    return ret
