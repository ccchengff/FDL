# -*- coding:utf-8 -*-

from collections import defaultdict
from functools import partial
import queue
try:
    import cPickle as pickle
except:
    import pickle

def make_topic_name(task_id, topic, src, dst, rank):
    if rank is None:
        return "{task_id}.{topic}.{src}.to.{dst}".format(
            task_id=task_id, topic=topic, 
            src=src, dst=dst)
    else:
        return "{task_id}.{topic}.{src}.to.{dst}.rank{rank}".format(
            task_id=task_id, topic=topic, 
            src=src, dst=dst, rank=rank)


class MessageChannelBase(object):
    """channel for message queue."""

    def __init__(self, task_id, my_party, rank=None, **kwargs):
        super(MessageChannelBase, self).__init__()
        self._task_id = task_id
        self._my_party = my_party
        self._rank = rank
        self._recv_queues = defaultdict(partial(queue.Queue, maxsize=100000))
        for name, value in kwargs.items():
            self.__setitem__(name, value)

    def connect(self):
        raise NotImplementedError
    
    def shutdown(self):
        raise NotImplementedError
    
    def send_async(self, content, dst_parties, topic):
        # do not serialize in background thread to 
        # avoid in place manipulation after the send method
        ser_content = self._serialize(content)
        self._do_send_async(ser_content, dst_parties, topic)

    def next_object(self, src_party, topic):
        # received contents are deserialized in background thread
        topic_name = make_topic_name(
            self._task_id, topic, src_party, self._my_party, self._rank)
        return self._recv_queues[topic_name].get()
    
    def _serialize(self, content):
        return pickle.dumps(content)
    
    def _deserialize(self, ser_content):
        return pickle.loads(ser_content)
    
    def _do_send_async(ser_content, dst_parties, topic):
        raise NotImplementedError
