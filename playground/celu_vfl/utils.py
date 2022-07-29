#-*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf
import numpy as np

from fdl.federation import Federation
from wdl_criteo import *
from dssm_avazu import *

import os, sys, traceback
import time
import logging
from collections import defaultdict

_DATASET_DIR = os.environ["DATASET_DIR"]

def load_data(args, shuffle=True):
    dataset_dir = os.path.join(_DATASET_DIR, args.data)
    train_path = os.path.join(dataset_dir, "train")
    valid_path = os.path.join(dataset_dir, "valid")
    train_path = train_path + ".npz"
    valid_path = valid_path + ".npz"
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
        keys = ["host"]
    
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
    model_fn = {}
    
    model_fn["vanilla"] = defaultdict(dict)
    model_fn["vanilla"]["criteo"] = wdl_criteo_fed
    model_fn["vanilla"]["avazu"] = dssm_avazu_fed

    model_fn["local_update"] = defaultdict(dict)
    model_fn["local_update"]["criteo"] = wdl_criteo_fed_with_local_update
    model_fn["local_update"]["avazu"] = dssm_avazu_fed_with_local_update

    if args.num_update_per_batch == 1:
        task_type = "vanilla"
    else:
        task_type = "local_update"
    
    fn = model_fn[task_type][args.data]
    return fn(args)

