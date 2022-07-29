#-*- coding:utf-8 -*-

import os
import time
import grpc
import logging

from fdl.federation import common_pb2 as common_pb

__DEFAULT_GRPC_CHANNEL_OPTIONS = (
    ('grpc.enable_http_proxy', 0),
    ('grpc.enable_https_proxy', 0),
    ('grpc.max_send_message_length', -1), 
    ('grpc.max_receive_message_length', -1)
)

def get_default_grpc_channel_options():
    return __DEFAULT_GRPC_CHANNEL_OPTIONS

def dict2pb(**kwargs):
    d = dict(kwargs)
    pb = []
    for k, v in d.items():
        assert isinstance(k, str) and isinstance(v, str), \
            "Key-value pairs should be provided by strings, got: ({}, {})".format(k, v)
        pb.append(common_pb.KVPair(key=k, value=v))
    return pb

def pb2dict(pb):
    return { kv.key: kv.value for kv in pb }

def make_ready_client(channel, stub_new, stop_event=None):
    channel_ready = grpc.channel_ready_future(channel)
    wait_secs = 0.5
    start_time = time.time()
    while (stop_event is None) or (not stop_event.is_set()):
        try:
            channel_ready.result(timeout=wait_secs)
            break
        except grpc.FutureTimeoutError:
            logging.warn("Channel not ready for {:.3f} seconds".format(
                         time.time() - start_time))
            if wait_secs < 5.0:
                wait_secs *= 1.2
        except (RuntimeError, SystemError, OSError, grpc.RpcError) as e:
            logging.warn("Waiting channel ready got: " + str(e))
    return stub_new(channel)

def rpc_with_retries(rpc_fn, failure_cb, description, 
                     rpc_lock=None, max_attempts=100):
    def func():
        num_attempts = 0
        while True:
            try:
                return rpc_fn()
            except grpc.RpcError as err:
                    num_attempts += 1
                    logging.warn(
                        "Exception in rpc call \"{}\" (#attempts: {}):\n{}".format(
                            description, num_attempts, err))
                    if num_attempts >= max_attempts:
                        logging.error(
                            "Failed after {} attempts, aborting...".format(
                                num_attempts))
                        traceback.print_exc()
                        os._exit(1)
                    elif failure_cb:
                        failure_cb()
                    else:
                        time.sleep(1) # sleep for one second by default

    if rpc_lock:
        with rpc_lock:
            return func()
    else:
        return func()
