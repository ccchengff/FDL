#-*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf

def label_infer_from_norm(x, neg_majority=True):
    x_norm = tf.math.sqrt(tf.reduce_sum(tf.math.pow(x, 2), axis=1))
    if not neg_majority:
        x_norm = -1 * x_norm
    return x_norm


def label_infer_from_direction(x, neg_majority=True):
    x_norm = tf.math.sqrt(tf.reduce_sum(tf.math.pow(x, 2), axis=1))
    normed_x = x / tf.reshape(x_norm + 1e-8, [-1, 1])
    cosines = tf.matmul(normed_x, normed_x, transpose_b=True)
    cosines = tf.clip_by_value(cosines, -1, 1)
    avg_cosines = tf.reduce_mean(cosines, axis=0)
    if neg_majority:
        avg_cosines = -1 * avg_cosines
    return avg_cosines


def label_infer_from_shorted_distance(x, pos_dev, neg_dev):
    pos_dis = tf.math.sqrt(tf.reduce_sum(tf.math.pow(x - pos_dev, 2), axis=1))
    neg_dis = tf.math.sqrt(tf.reduce_sum(tf.math.pow(x - neg_dev, 2), axis=1))
    return tf.cast(pos_dis < neg_dis, tf.float32)
