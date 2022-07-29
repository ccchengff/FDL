# -*- coding:utf-8 -*-

import os, sys
import time
import grpc
from concurrent import futures
import threading
from collections import defaultdict
import queue
import logging

from fdl.federation.channel.message_channel_base import MessageChannelBase, make_topic_name
from fdl.federation.channel import message_channel_pb2 as mc_pb
from fdl.federation.channel import message_channel_pb2_grpc as mc_grpc
from fdl.federation import common_pb2 as common_pb
from fdl.utils import grpc_util

class GrpcMessageChannel(MessageChannelBase):
    """gRPC-based channel for message queue."""

    class _MessageChannelService(mc_grpc.MessageChannelServiceServicer):
        
        def __init__(self, 
                     connect_channel_handler, 
                     send_msg_handler, 
                     shutdown_channel_handler):
            super(GrpcMessageChannel._MessageChannelService, self).__init__()
            self._connect_channel_handler = connect_channel_handler
            self._send_msg_handler = send_msg_handler
            self._shutdown_channel_handler = shutdown_channel_handler
        
        def ConnectChannel(self, request, context):
            return self._connect_channel_handler(request, context)
        
        def SendMsg(self, request, context):
            return self._send_msg_handler(request, context)
        
        def ShutdownChannel(self, request, context):
            return self._shutdown_channel_handler(request, context)

    def __init__(self, task_id, my_party, rank=None, sugguested_port=0, **kwargs):
        super(GrpcMessageChannel, self).__init__(task_id, my_party, rank=rank, **kwargs)
        self._sugguested_port = sugguested_port
        self._is_connected = False
        self._init_server()
    
    def connect(self, party2address):
        # connect to other parties
        self._party2address = party2address
        self._participants = set(self._party2address.keys()) - set((self._my_party,))
        self._grpc_channels = {}
        self._clients = {}
        
        for party in self._participants:
            addr = self._party2address[party]
            logging.info(f"Connecting to party[{party}] address[{addr}]...")
            self._init_client(party, addr)
            response = grpc_util.rpc_with_retries(
                lambda: self._clients[party].ConnectChannel(
                    mc_pb.ConnectChannelRequest(
                        task_id=self._task_id, 
                        party=self._my_party, 
                        rank=self._rank)), 
                failure_cb=(lambda : self._init_client(party, addr)), 
                description=f"ConnectChannel-{party}")
            if response.status != common_pb.SUCCESS:
                raise RuntimeError(
                    f"Failed to connect to party[{party}], " + 
                    f"error msg: {response.err_msg}")
            logging.info(f"Connected to party[{party}] successfully")
        
        # send msg ids and futures
        self._send_msg_ids = defaultdict(lambda: defaultdict(int))
        self._send_futures = queue.Queue(1000000)

        def monitor_fn():
            while True:
                response_future, request = self._send_futures.get()
                if response_future is None:
                    break
                response = response_future.result()
                if response.status != common_pb.SUCCESS:
                    # TODO: resend the message
                    msg = f"Failed to send message[{request.msg_id}] " + \
                        f"of topic[{request.topic_name}], " + \
                        f"error msg: {response.err_msg}"
                    logging.error(msg)
                    import _thread
                    _thread.interrupt_main() # raise a KeyboardInterrupt in main thread
                    break
            
        self._monitor_thread = threading.Thread(target=monitor_fn, daemon=True)
        self._monitor_thread.start()
        self._is_connected = True
    
    def shutdown(self):
        logging.info("Closing channels to other parties...")
        
        if self._is_connected:
            self._send_futures.put((None, None))
            self._monitor_thread.join()
            for party in self._participants:
                response = grpc_util.rpc_with_retries(
                    lambda: self._clients[party].ShutdownChannel(
                        mc_pb.ShutdownChannelRequest(
                            task_id=self._task_id, 
                            party=self._my_party, 
                            rank=self._rank)), 
                    failure_cb=self._init_client, 
                    description=f"ShutdownChannel-{party}")
                if response.status != common_pb.SUCCESS:
                    raise RuntimeError(
                        f"Failed to shutdown channel to party[{party}], " + 
                        f"error msg: {response.err_msg}")
            self._is_connected = False
        
        if self._is_service_started:
            cnt = 0
            while True:
                waiting_parties = self._served_parties - self._can_stop_parties
                if len(waiting_parties) > 0:
                    cnt += 1
                    if cnt % 10 == 0: # warn every 10 seconds
                        logging.warn(
                            "Waiting for shutdown request from " + 
                            str(waiting_parties))
                    time.sleep(1)
                else:
                    self._server.stop(1)
                    break
            self._is_service_started = False
        
        logging.info("Closed channels successfully")

    def _init_server(self):
        # receive msg ids and unordered buffers
        self._recv_msg_ids = defaultdict(int)
        self._unordered_recv_buffers = defaultdict(dict)
        self._recv_conditions = defaultdict(threading.Condition)
        self._served_parties = set()
        self._can_stop_parties = set()
        
        # start service
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10), 
            maximum_concurrent_rpcs=100, 
            options=grpc_util.get_default_grpc_channel_options())
        mc_grpc.add_MessageChannelServiceServicer_to_server(
            GrpcMessageChannel._MessageChannelService(
                self._connect_channel_handler, 
                self._send_msg_handler, 
                self._shutdown_channel_handler), 
            self._server)
        self._port = self._server.add_insecure_port(
            f"[::]:{self._sugguested_port}")
        self._server.start()
        self._is_service_started = True
        if self._sugguested_port != 0 and self._port != self._sugguested_port:
            logging.warn(
                f"Suggested port {self._sugguested_port} is not available, " + 
                f"binded on port {self._port} instead")
        logging.info(f"Started service on port {self._port}")
    
    def _init_client(self, target_party, target_address):
        assert target_party != self._my_party
        if target_party in self._grpc_channels:
            self._grpc_channels[target_party].close()
        self._grpc_channels[target_party] = grpc.insecure_channel(
            target_address,
            options=grpc_util.get_default_grpc_channel_options())
        self._clients[target_party] = grpc_util.make_ready_client(
            self._grpc_channels[target_party], 
            mc_grpc.MessageChannelServiceStub)

    def _do_send_async(self, ser_content, dst_parties, topic):
        for dst in dst_parties:
            topic_name = make_topic_name(
                self._task_id, topic, self._my_party, dst, self._rank)
            request = mc_pb.MsgRequest(
                topic_name=topic_name, 
                msg_id=self._send_msg_ids[dst][topic], 
                content=ser_content)
            logging.debug(
                f"Sending message[{request.msg_id}] " + 
                f"of topic[{request.topic_name}] " + 
                f"with {len(request.content)} bytes")
            
            def cb(response_future):
                response = response_future.result()
                if response.status != common_pb.SUCCESS:
                    msg = f"Failed to send message[{request.msg_id}] " + \
                        f"of topic[{request.topic_name}], " + \
                        f"error msg: {response.err_msg}"
                    logging.error()

            response_future = self._clients[dst].SendMsg.future(request)
            response_future.add_done_callback(lambda x : None)
            self._send_futures.put((response_future, request))
            self._send_msg_ids[dst][topic] += 1
    
    def _connect_channel_handler(self, request, context):
        logging.info(
            f"Received connect channel request: task_id[{request.task_id}] " + 
            f"party[{request.party}] rank[{request.rank}]")
        if request.task_id != self._task_id:
            err_msg = "Invalid task id: " + request.task_id
            logging.warn(err_msg)
            return common_pb.Response(status=common_pb.ERROR, err_msg=err_msg)
        else:
            self._served_parties.add(request.party)
            return common_pb.Response(status=common_pb.SUCCESS)

    def _send_msg_handler(self, request, context):
        logging.debug(
            f"Received message[{request.msg_id}] " + 
            f"of topic[{request.topic_name}] " + 
            f"with {len(request.content)} bytes")
        
        msg_id = request.msg_id
        topic_name = request.topic_name
        ser_content = request.content
        content = self._deserialize(ser_content)
        with self._recv_conditions[topic_name]:
            expected_recv_id = self._recv_msg_ids[topic_name]
            if msg_id != expected_recv_id:
                self._unordered_recv_buffers[topic_name][msg_id] = content
            else:
                self._recv_queues[topic_name].put(content)
                expected_recv_id += 1
                while expected_recv_id in self._unordered_recv_buffers[topic_name]:
                    self._recv_queues[topic_name].put(
                        self._unordered_recv_buffers[topic_name].pop(
                            expected_recv_id))
                    expected_recv_id += 1
                self._recv_msg_ids[topic_name] = expected_recv_id

        return common_pb.Response(status=common_pb.SUCCESS)
    
    def _shutdown_channel_handler(self, request, context):
        logging.info(
            f"Received shutdown channel request: task_id[{request.task_id}] " + 
            f"party[{request.party}] rank[{request.rank}]")
        if request.task_id != self._task_id:
            err_msg = "Invalid task id: " + request.task_id
            logging.warn(err_msg)
            return common_pb.Response(status=common_pb.ERROR, err_msg=err_msg)
        elif request.party not in self._served_parties:
            err_msg = f"Party[{request.party}] has not connected yet"
            logging.warn(err_msg)
            return common_pb.Response(status=common_pb.ERROR, err_msg=err_msg)
        else:
            self._can_stop_parties.add(request.party)
            return common_pb.Response(status=common_pb.SUCCESS)
    
    @property
    def port(self):
        return self._port
