#-*- coding:utf-8 -*-

import dataclasses
import yaml
import logging

@dataclasses.dataclass
class TaskConfig:
    task_id: str
    my_party: str
    ip: str
    rank: int
    channel_type: str
    scheduler_config: dict = dataclasses.field(default_factory=dict)
    parties: dict = dataclasses.field(default_factory=dict)
    
    @property
    def all_parties(self):
        return sorted(list(self.parties.keys()))
    
    @property
    def other_parties(self):
        return sorted(list(filter(lambda x: x != self.my_party, self.parties.keys())))

    @property
    def num_parties(self):
        return len(self.parties)

class Federation(object):
    """Static objects"""
    _task_config = None
    _scheduler_client = None
    _msg_channel = None

    @staticmethod
    def init_federation(task_id, party, config_file, ip=None, rank=0):
        logging.info("Initializing federation...")

        task_config = Federation._init_task_config(
            task_id, party, config_file, 
            ip=ip, rank=rank)

        scheduler_client = Federation._init_scheduler_client(task_config)
        msg_channel = Federation._init_msg_channel(task_config, scheduler_client)
        
        Federation._task_config = task_config
        Federation._scheduler_client = scheduler_client
        Federation._msg_channel = msg_channel
        
        # shake
        logging.debug("Shaking with other parties...")
        other_parties = task_config.other_parties
        Federation.send_async(task_config.my_party, dst_parties=other_parties, topic="shake")
        for party in other_parties:
            logging.debug(f"Shaking with {party}...")
            msg = Federation.next_object(src_party=party, topic="shake")
            if msg == party:
                logging.debug(f"Shaked with {party} successfully")
            else:
                err_msg = f"Received incorrect message when shaking with {party}: {msg}"
                logging.error(err_msg)
                raise RuntimeError(err_msg)
        logging.debug("Shaked with other parties successfully")

        logging.info("Initialization of federation done")
    
    @staticmethod
    def _init_task_config(task_id, my_party, config_file, ip=None, rank=0):
        with open(config_file) as stream:
            loaded_config = yaml.safe_load(stream)
        task_config = TaskConfig(
            task_id=task_id, 
            my_party=my_party, 
            ip=ip, 
            rank=rank, 
            channel_type=loaded_config.get("channel_type", "grpc"), 
            scheduler_config=loaded_config.get("scheduler_config", {}))
        parties_json = loaded_config["parties"]
        for party, loaded_party_config in loaded_config["parties"].items():
            task_config.parties[party] = loaded_party_config or {}
        return task_config
    
    @staticmethod
    def _init_scheduler_client(task_config):
        scheduler_config = task_config.scheduler_config
        scheduler_type = scheduler_config.get("type", None)
        if not scheduler_type:
            return None
        
        scheduler_address = scheduler_config["address"]
        scheduler_token = scheduler_config.get("token", None)
        if scheduler_type == "grpc":
            from fdl.federation.scheduler.grpc_scheduler import GrpcSchedulerClient
            scheduler_client = GrpcSchedulerClient(
                task_config.task_id, 
                task_config.my_party, 
                scheduler_address, 
                rank=task_config.rank, 
                server_token=scheduler_token)
        elif scheduler_type == "pulsar":
            from fdl.federation.scheduler.pulsar_scheduler import PulsarSchedulerClient
            topic_prefix = scheduler_config.get("topic_prefix", None)
            scheduler_client = PulsarSchedulerClient(
                task_config.task_id, 
                task_config.my_party, 
                scheduler_address, 
                rank=task_config.rank, 
                pulsar_token=scheduler_token, 
                topic_prefix=topic_prefix)
        else:
            raise ValueError(f"No such scheduler type: {scheduler_type}")
        return scheduler_client
    
    @staticmethod
    def _init_msg_channel(task_config, scheduler_client):
        channel_type = task_config.channel_type
        if channel_type == "grpc":
            from fdl.federation.channel.grpc_message_channel import GrpcMessageChannel
            party2address = { 
                party: party_config["channel_address"] \
                for party, party_config in task_config.parties.items() \
                if "channel_address" in party_config }
            if scheduler_client is None and len(party2address) != task_config.num_parties:
                missing = set(task_config.all_parties) - set(party2address.keys())
                raise ValueError(
                    "Must provide grpc channel addresses for all parties "
                    f"when scheduler is none, missing {missing}")
            
            if task_config.my_party in party2address:
                sugguested_port = int(party2address[task_config.my_party].split(':')[1])
            else:
                sugguested_port = 0
            msg_channel = GrpcMessageChannel(
                task_config.task_id, 
                task_config.my_party, 
                rank=task_config.rank, 
                sugguested_port=sugguested_port)
            
            if scheduler_client is not None:
                # register grpc server address to scheduler
                scheduler_client.register_trainer(
                    party=task_config.my_party, 
                    rank=task_config.rank, 
                    channel_address=f"{task_config.my_ip}:{msg_channel.port}")
                # fetch grpc server addresses of other parties
                for party in task_config.parties.keys():
                    if party == task_config.my_party:
                        continue
                    tmp = scheduler_client.get_trainer_config(party)
                    if tmp["party"] != party and tmp["rank"] != task_config.rank:
                        raise RuntimeError(
                            "Fetched channel address of "
                            f"party[{tmp['party']}] rank[{tmp['rank']}], "
                            f"expeceted party[{party}] rank[{rank}]")
                    task_config.parties[party]["channel_address"] = tmp["channel_address"]
            else:
                if msg_channel.port != sugguested_port:
                    raise RuntimeError(
                        f"Channel service must bind to the sugguested port["
                        f"{sugguested_port}] when scheduler is none")
        
            party2address = { 
                party: party_config["channel_address"] \
                for party, party_config in task_config.parties.items() }
            msg_channel.connect(party2address)
        else:
            raise ValueError(f"No such channel type: {channel_type}")
        return msg_channel
    
    @staticmethod
    def shutdown_federation():
        if Federation._msg_channel is not None:
            Federation._msg_channel.shutdown()
        if Federation._scheduler_client is not None:
            Federation._scheduler_client.shutdown()
    
    @staticmethod
    def send_async(content, dst_parties="auto", topic=None):
        if dst_parties == "auto":
            dst_parties = [Federation.get_peer_party()]
        elif isinstance(dst_parties, str):
            dst_parties = [dst_parties]
        Federation._msg_channel.send_async(
            content, 
            dst_parties=dst_parties, 
            topic=(topic or "default"))
    
    @staticmethod
    def next_object(src_party="auto", topic=None):
        if src_party == "auto":
            src_party = Federation.get_peer_party()
        return Federation._msg_channel.next_object(
            src_party, 
            topic=(topic or "default"))
    
    @staticmethod
    def sync(parties="auto"):
        if parties == "auto":
            parties = [Federation.get_peer_party()]
        elif isinstance(parties, str):
            parties = [parties]
        sync_topic = "default_sync"
        Federation._msg_channel.send_async(
            True, 
            dst_parties=parties, 
            topic=sync_topic)
        for party in parties:
            Federation._msg_channel.next_object(
                party, 
                topic=sync_topic)
    
    @staticmethod
    def party2string(parties):
        if parties == "auto":
            return Federation.get_peer_party()
        elif isinstance(parties, str):
            return parties
        else:
            return ",".join(list(parties))
    
    @staticmethod
    def get_all_parties():
        return Federation._task_config.all_parties
    
    @staticmethod
    def get_other_parties():
        return Federation._task_config.other_parties
    
    @staticmethod
    def get_num_parties():
        return Federation._task_config.num_parties
    
    @staticmethod
    def get_peer_party():
        if Federation._task_config.num_parties > 2:
            raise RuntimeError("Cannot get peer party when there are more than two parties")
        return Federation._task_config.other_parties[0]
