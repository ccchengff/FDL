#-*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf
import numpy as np

import os, sys, traceback
import logging

def dssm_movielens_guest_bottom(guest_input, device="/gpu:0", num_buckets=5000, embedding_size=128):
    with tf.variable_scope("guest_bottom", dtype=tf.float32, reuse=tf.AUTO_REUSE, 
                           initializer=tf.glorot_normal_initializer()):
        with tf.device(device):
            embedding = tf.get_variable(name="Embedding", shape=(num_buckets, embedding_size))
            dim = guest_input.shape[1] * embedding_size
            W2 = tf.get_variable(name='W2', shape=[dim, 256])
            W3 = tf.get_variable(name='W3', shape=[256, 256])

            act1 = tf.nn.embedding_lookup(embedding, guest_input)
            act1 = tf.reshape(act1, (-1, dim))
            act2 = tf.nn.dropout(tf.nn.relu(tf.matmul(act1, W2)), 0.5)
            act3 = tf.nn.dropout(tf.matmul(act2, W3), 0.5)

            return act3


def dssm_movielens_host_bottom(host_input, device="/gpu:0", num_buckets=5000, embedding_size=128):
    with tf.variable_scope("host_bottom", dtype=tf.float32, reuse=tf.AUTO_REUSE, 
                           initializer=tf.glorot_normal_initializer()):
        with tf.device(device):
            embedding = tf.get_variable(name="Embedding", shape=(num_buckets, embedding_size))
            dim = host_input.shape[1] * embedding_size
            W2 = tf.get_variable(name='W2', shape=[dim, 256])
            W3 = tf.get_variable(name='W3', shape=[256, 256])

            act1 = tf.nn.embedding_lookup(embedding, host_input)
            act1 = tf.reshape(act1, (-1, dim))
            act2 = tf.nn.dropout(tf.nn.relu(tf.matmul(act1, W2)), 0.5)
            act3 = tf.nn.dropout(tf.matmul(act2, W3), 0.5)

            return act3


def dssm_movielens_top_model(labels, guest_output, host_output, device="/gpu:0"):
    with tf.variable_scope("top", dtype=tf.float32, reuse=tf.AUTO_REUSE, 
                           initializer=tf.glorot_normal_initializer()):
        with tf.device(device):
            dot_product = tf.multiply(guest_output, host_output)
            logits = tf.reshape(tf.reduce_sum(dot_product, axis=1), [-1, 1])
            proba = tf.nn.sigmoid(logits)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labels))
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
