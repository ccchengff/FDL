#-*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf
import numpy as np

import os, sys, traceback
import logging

def wdl_criteo_guest_bottom(guest_input, device="/gpu:0"):
    with tf.variable_scope("guest_bottom", dtype=tf.float32, reuse=tf.AUTO_REUSE, 
                           initializer=tf.random_normal_initializer(stddev=0.01)):
        with tf.device(device):
            dim = guest_input.shape[1]
            W1 = tf.get_variable(name='W1', shape=[dim, 256])
            W2 = tf.get_variable(name='W2', shape=[256, 256])
            W3 = tf.get_variable(name='W3', shape=[256, 256])

            act1 = tf.nn.dropout(tf.nn.relu(tf.matmul(guest_input, W1)), 0.5)
            act2 = tf.nn.dropout(tf.nn.relu(tf.matmul(act1, W2)), 0.5)
            act3 = tf.nn.dropout(tf.matmul(act2, W3), 0.5)

            return act3


def wdl_criteo_host_bottom(host_input, device="/gpu:0", num_buckets=1000000, embedding_size=128):
    with tf.variable_scope("host_bottom", dtype=tf.float32, reuse=tf.AUTO_REUSE, 
                           initializer=tf.random_normal_initializer(stddev=0.01)):
        with tf.device(device):
            embedding = tf.get_variable(name="Embedding", shape=(num_buckets, embedding_size))
            dim = host_input.shape[1] * embedding_size
            W2 = tf.get_variable(name='W2', shape=[dim, 256])
            W3 = tf.get_variable(name='W3', shape=[256, 256])
            
            hash_buckets = tf.strings.to_hash_bucket_fast(
                tf.strings.as_string(host_input), num_buckets)
            act1 = tf.nn.embedding_lookup(embedding, hash_buckets)
            act1 = tf.reshape(act1, (-1, dim))
            act2 = tf.nn.dropout(tf.nn.relu(tf.matmul(act1, W2)), 0.5)
            act3 = tf.nn.dropout(tf.matmul(act2, W3), 0.5)

            return act3


def wdl_criteo_top_model(labels, guest_output, host_output, device="/gpu:0"):
    with tf.variable_scope("top", dtype=tf.float32, reuse=tf.AUTO_REUSE, 
                           initializer=tf.random_normal_initializer(stddev=0.01)):
        with tf.device(device):
            concat = tf.concat((guest_output, host_output), 1)
            W1 = tf.get_variable(name='W1', shape=[256 * 2, 1])
            
            logits = tf.matmul(concat, W1)
            proba = tf.nn.sigmoid(logits)
            loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                logits=logits, labels=labels, pos_weight=3.0))
            corrects = tf.math.equal(tf.cast(logits >= 0, tf.float32), labels)
            acc = tf.reduce_mean(tf.cast(corrects, tf.float32))

            pos_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                logits=logits, labels=tf.ones_like(logits), pos_weight=3.0))
            neg_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                logits=logits, labels=tf.zeros_like(logits), pos_weight=3.0))

            return {
                "logits": logits, 
                "proba": proba, 
                "loss": loss, 
                "acc": acc, 
                "pos_loss": pos_loss, 
                "neg_loss": neg_loss, 
            }
