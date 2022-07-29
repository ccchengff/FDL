#-*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf
from fdl.federation import Federation
import logging

def send_op(name, tensor, dst_parties="auto", op_name=None):
    def func(x):
        Federation.send_async(x.numpy(), dst_parties=dst_parties, topic=name)
        return x
    
    if not op_name:
        op_name = name + ".SendTo." + Federation.party2string(dst_parties)
    return tf.py_function(func=func, inp=[tensor], Tout=[tensor.dtype], 
                          name=op_name)[0]

def recv_op(name, src_party="auto", dtype=tf.float32, shape=None, op_name=None):
    def func():
        x = Federation.next_object(src_party=src_party, topic=name)
        x = tf.convert_to_tensor(x, dtype=dtype)
        return x
    
    if not op_name:
        op_name = name + ".RecvFrom." + Federation.party2string(src_party)
    tensor = tf.py_function(func=func, inp=[], Tout=[dtype])[0]
    if shape:
        tensor = tf.ensure_shape(tensor, shape, name=op_name)
    else:
        tensor = tf.identity(tensor, name=op_name)
    return tensor
