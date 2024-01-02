#-*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf
import numpy as np

from fdl.federation import Federation
from fdl.tensorflow.vfl_graph import FLGuestGraph, FLHostGraph
from fdl.tensorflow.comm_ops import send_op, recv_op
from fdl.tensorflow.privacy.proj_pert import projection_perturb_fn
from fdl.tensorflow.privacy.marvell import marvell_perturb_fn
from fdl.tensorflow.privacy.label_inference import label_infer_from_norm
from fdl.tensorflow.privacy.label_inference import label_infer_from_direction
from fdl.tensorflow.privacy.label_inference import label_infer_from_shorted_distance

import os, sys, traceback
import time
import logging
from collections import defaultdict

_DATASET_DIR = os.environ["DATASET_DIR"]

def load_data(args, shuffle=True):
    dataset_dir = os.path.join(_DATASET_DIR, args.data)
    train_path = os.path.join(dataset_dir, "train.npz")
    valid_path = os.path.join(dataset_dir, "valid.npz")
    logging.info("Train path: " + train_path)
    logging.info("Valid path: " + valid_path)

    if shuffle:
        if args.party == "guest":
            random_seed = int(time.time())
            Federation.send_async(random_seed)
        else:
            random_seed = Federation.next_object()
        rand_state = np.random.RandomState(seed=random_seed)
        train_perm = None
        valid_perm = None
    
    if args.party == "guest":
        keys = ["guest", "labels"]
    else:
        # Here we let host load labels for 
        # the evaluation of label inference methods
        keys = ["host", "labels"]
    
    train_loader = np.load(train_path)
    train = []
    for k in keys:
        arr = train_loader[k]
        if shuffle:
            if train_perm is None:
                train_perm = rand_state.permutation(arr.shape[0])
            arr = arr[train_perm]
        train.append(arr)
        logging.info(f"Train {k} shape: {train[-1].shape}")
    
    valid_loader = np.load(valid_path)
    valid = []
    for k in keys:
        arr = valid_loader[k]
        if shuffle:
            if valid_perm is None:
                valid_perm = rand_state.permutation(arr.shape[0])
            arr = arr[valid_perm]
        valid.append(arr)
        logging.info(f"Valid {k} shape: {valid[-1].shape}")
    
    return tuple(train + valid)


def define_model(args):
    if args.data == "criteo":
        from wdl_criteo import wdl_criteo_guest_bottom as guest_bottom_fn
        from wdl_criteo import wdl_criteo_host_bottom as host_bottom_fn
        from wdl_criteo import wdl_criteo_top_model as top_model_fn
        guest_input = tf.placeholder(tf.float32, [None, 13], name="guest-input")
        host_input = tf.placeholder(tf.int32, [None, 26], name="host-input")
        labels = tf.placeholder(tf.float32, [None, 1], name="labels")
    elif args.data == "avazu":
        from wdl_avazu import wdl_avazu_guest_bottom as guest_bottom_fn
        from wdl_avazu import wdl_avazu_host_bottom as host_bottom_fn
        from wdl_avazu import wdl_avazu_top_model as top_model_fn
        guest_input = tf.placeholder(tf.int32, [None, 8], name="guest-input")
        host_input = tf.placeholder(tf.int32, [None, 14], name="host-input")
        labels = tf.placeholder(tf.float32, [None, 1], name="labels")
    elif args.data == "movielens_1m":
        from dssm_movielens import dssm_movielens_guest_bottom as guest_bottom_fn
        from dssm_movielens import dssm_movielens_host_bottom as host_bottom_fn
        from dssm_movielens import dssm_movielens_top_model as top_model_fn
        guest_input = tf.placeholder(tf.int32, [None, 3], name="guest-input")
        host_input = tf.placeholder(tf.int32, [None, 4], name="host-input")
        labels = tf.placeholder(tf.float32, [None, 1], name="labels")
    else:
        raise ValueError(f"No such data: {args.data}")

    optimizer = tf.train.AdagradOptimizer(args.eta)
    if args.party == "guest":
        graph = FLGuestGraph(deps=[guest_input])
        # define bottom model of guest
        guest_act = guest_bottom_fn(guest_input, device=args.device)
        # receive the output of bottom model of host
        host_act = graph.remote_bottom("HostAct", shape=[None, 256])
        # define top model of guest
        top_model = top_model_fn(labels, guest_act, host_act, device=args.device)
        # minimize, perturb and send derivatives
        pos_dev = tf.gradients(top_model["pos_loss"], host_act)[0]
        neg_dev = tf.gradients(top_model["neg_loss"], host_act)[0]
        perturb_fn = get_perturb_fn(args, labels, pos_dev, neg_dev)
        train_op = graph.minimize(
            optimizer, top_model["loss"], 
            perturb_fn=perturb_fn, 
            return_grads=False)
        pred_op = graph.predict(top_model["logits"])
        return guest_input, labels, top_model, train_op, pred_op
    else:
        graph = FLHostGraph()
        # define bottom model of host
        host_act = host_bottom_fn(host_input, device=args.device)
        # send the output of bottom model of host
        graph.send_bottom("HostAct", host_act)
        # receive derivatives, minimize
        train_op, _, devs_and_acts = graph.minimize(optimizer, return_grads=True)
        pred_op = graph.predict()
        # infer labels via derivatives
        host_dev = list(filter(lambda x: x[1] is host_act, devs_and_acts))[0][0]
        infer_logits = []
        infer_fns = get_infer_fns()
        if len(infer_fns) > 0:
            for infer_name, infer_fn in infer_fns:
                infer_logits.append((infer_name, infer_fn(host_dev)))
        return host_input, train_op, pred_op, infer_logits


def get_infer_fns():
    infer_fns = [
        ("Norm", label_infer_from_norm), 
        ("Direction", label_infer_from_direction), 
        ]
    return infer_fns


def get_perturb_fn(args, labels, pos_dev, neg_dev):
    if not args.perturb or args.perturb == "none":
        return None
    elif args.perturb == "iso-proj":
        return lambda dev: projection_perturb_fn(
            dev, labels, sum_kl_bound=args.sum_kl_bound, iso_proj=True)
    elif args.perturb == "proj":
        return lambda dev: projection_perturb_fn(
            dev, labels, sum_kl_bound=args.sum_kl_bound, iso_proj=False)
    elif args.perturb == "marvell":
        return lambda dev: marvell_perturb_fn(
            dev, labels, init_scale=args.init_scale)
    else:
        raise ValueError(f"No such perturbance method: {args.perturb}")
