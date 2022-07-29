#-*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf
import numpy as np

from fdl.federation import Federation
from fdl.tensorflow.vfl_graph import FLGuestGraph, FLHostGraph

import os, sys, traceback
import logging

def wdl_criteo_fed(args):
    optimizer = tf.train.AdagradOptimizer(args.eta)
    if args.party == "guest":
        guest_input = tf.placeholder(tf.float32, [None, 13], name="guest-input")
        labels = tf.placeholder(tf.float32, [None, 1], name="labels")
        graph = FLGuestGraph(deps=[guest_input])
        # bottom model of guest
        guest_act = wdl_criteo_guest_bottom(guest_input, device=args.device)
        # top model & minimize
        host_act = graph.remote_bottom("HostAct", shape=[None, 256])
        top_model = wdl_criteo_top_model(labels, guest_act, host_act, device=args.device)
        train_op, _, devs_and_acts = graph.minimize(
            optimizer, top_model["loss"], 
            return_grads=True)
        pred_op = graph.predict(top_model["logits"])
        return guest_input, labels, top_model, train_op, pred_op
    else:
        host_input = tf.placeholder(tf.int32, [None, 26], name="host-input")
        graph = FLHostGraph()
        host_act = wdl_criteo_host_bottom(host_input, device=args.device)
        graph.send_bottom("HostAct", host_act)
        train_op, _, devs_and_acts = graph.minimize(optimizer, return_grads=True)
        pred_op = graph.predict()
        return host_input, train_op, pred_op


def wdl_criteo_fed_with_local_update(args):
    optimizer = tf.train.AdagradOptimizer(args.eta)
    if args.party == "guest":
        guest_input = tf.placeholder(tf.float32, [None, 13], name="guest-input")
        labels = tf.placeholder(tf.float32, [None, 1], name="labels")
        graph = FLGuestGraph(deps=[guest_input])
        # bottom model of guest
        guest_act = wdl_criteo_guest_bottom(guest_input, device=args.device)
        # top model & minimize
        host_act = graph.remote_bottom("HostAct", shape=[None, 256])
        top_model = wdl_criteo_top_model(labels, guest_act, host_act, device=args.device)
        train_op, _, devs_and_acts = graph.minimize(
            optimizer, top_model["loss"], 
            return_grads=True)
        pred_op = graph.predict(top_model["logits"])
        # local update
        host_dev = list(filter(lambda x: x[1] is host_act, devs_and_acts))[0][0]
        cache_host_act = tf.placeholder(tf.float32, host_act.shape, name="CacheAct")
        cache_host_dev = tf.placeholder(tf.float32, host_dev.shape, name="CacheDev")
        local_top_model = wdl_criteo_top_model(
            labels, guest_act, cache_host_act, 
            device=args.device)
        local_train_op, ins_weights = graph.local_minimize(
            optimizer, local_top_model["ins_loss"], 
            {"HostAct" : cache_host_act}, 
            {"HostAct" : cache_host_dev}, 
            args.sim_thres)
        return guest_input, labels, top_model, train_op, pred_op, \
            host_act, host_dev, cache_host_act, cache_host_dev, \
            local_top_model, local_train_op
    else:
        host_input = tf.placeholder(tf.int32, [None, 26], name="host-input")
        graph = FLHostGraph()
        host_act = wdl_criteo_host_bottom(host_input, device=args.device)
        graph.send_bottom("HostAct", host_act)
        train_op, _, devs_and_acts = graph.minimize(optimizer, return_grads=True)
        pred_op = graph.predict()
        # local update
        host_dev = list(filter(lambda x: x[1] is host_act, devs_and_acts))[0][0]
        cache_host_act = tf.placeholder(tf.float32, host_act.shape, name="CacheAct")
        cache_host_dev = tf.placeholder(tf.float32, host_dev.shape, name="CacheDev")
        local_train_op, ins_weights = graph.local_minimize(
            optimizer, 
            {"HostAct" : cache_host_dev}, 
            {"HostAct" : cache_host_act}, 
            args.sim_thres)
        return host_input, train_op, pred_op, \
            host_act, host_dev, cache_host_act, cache_host_dev, \
            local_train_op


def wdl_criteo_guest_bottom(guest_input, device="/gpu:0"):
    with tf.variable_scope("guest_bottom", dtype=tf.float32, 
                           initializer=tf.random_normal_initializer(stddev=0.01)):
        with tf.device(device):
            dim = guest_input.shape[1]
            W1 = tf.get_variable(name='W1', shape=[dim, 256])
            W2 = tf.get_variable(name='W2', shape=[256, 256])
            W3 = tf.get_variable(name='W3', shape=[256, 256])

            act1 = tf.nn.relu(tf.matmul(guest_input, W1))
            act2 = tf.nn.relu(tf.matmul(act1, W2))
            act3 = tf.matmul(act2, W3)

            return act3


def wdl_criteo_host_bottom(host_input, device="/gpu:0", num_buckets=1000000, embedding_size=128):
    with tf.variable_scope("host_bottom", dtype=tf.float32, 
                           initializer=tf.random_normal_initializer(stddev=0.01)):
        with tf.device(device):
            embedding = tf.get_variable(name="Embedding", shape=(num_buckets, embedding_size))
            dim = host_input.shape[1] * embedding_size
            W2 = tf.get_variable(name='W1', shape=[dim, 256])
            W3 = tf.get_variable(name='W2', shape=[256, 256])
            
            hash_buckets = tf.strings.to_hash_bucket_fast(
                tf.strings.as_string(host_input), num_buckets)
            act1 = tf.nn.embedding_lookup(embedding, hash_buckets)
            act1 = tf.reshape(act1, (-1, dim))
            act2 = tf.nn.relu(tf.matmul(act1, W2))
            act3 = tf.matmul(act2, W3)

            return act3


def wdl_criteo_top_model(labels, guest_output, host_output, device="/gpu:0"):
    with tf.variable_scope("top", dtype=tf.float32, reuse=tf.AUTO_REUSE, 
                           initializer=tf.random_normal_initializer(stddev=0.01)):
        with tf.device(device):
            concat = tf.concat((guest_output, host_output), 1)
            W1 = tf.get_variable(name='W1', shape=[256 * 2, 1])
            
            logits = tf.matmul(concat, W1)
            proba = tf.nn.sigmoid(logits)
            ins_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labels)
            loss = tf.reduce_mean(ins_loss)
            corrects = tf.math.equal(tf.cast(logits >= 0, tf.float32), labels)
            acc = tf.reduce_mean(tf.cast(corrects, tf.float32))

            return {
                "logits": logits, 
                "proba": proba, 
                "ins_loss": ins_loss, 
                "loss": loss, 
                "acc": acc, 
            }

