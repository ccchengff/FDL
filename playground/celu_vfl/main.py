#-*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf
import numpy as np
from sklearn import metrics

from fdl.federation import Federation
from utils import load_data, define_model

import os, sys, traceback
import time
import random
import argparse
import threading
import logging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=str, default="test_task", help="Task id")
    parser.add_argument("--party", type=str, default="guest", help="Party name")
    parser.add_argument("--config-file", type=str, default="config.yaml", help="Config yaml")
    parser.add_argument("--data", type=str, choices=("criteo", "avazu"), help="Name of dataset")
    parser.add_argument("--device", type=str, default="/gpu:0", help="Name of device")
    parser.add_argument("--epochs", type=int, default=5, help="Max number of epochs")
    parser.add_argument("--eta", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    parser.add_argument("--valid-freq", type=int, default=2000, help="Frequency of evaluation on valid set")
    parser.add_argument("--print-freq", type=int, default=20, help="Frequency of printing metrics")
    parser.add_argument("--num-batch-per-workset", type=int, default=5, help="Number of batch per workset")
    parser.add_argument("--num-update-per-batch", type=int, default=5, help="Number of updates per batch")
    parser.add_argument("--sim-thres", type=float, default=0.5, help="Threshold for similarity")
    
    args = parser.parse_args()
    assert args.party in ("guest", "host")
    logging.info(f"Args: {args}")
    return args


def celu_vfl_main(args):
    global num_total_comms, num_local_updates
    
    # define model
    model = define_model(args)
    if args.party == "guest":
        guest_input, labels, top_model, train_op, pred_op, \
            host_act, host_dev, cache_host_act, cache_host_dev, \
            local_top_model, local_train_op = model
    else:
        host_input, train_op, pred_op, \
            host_act, host_dev, cache_host_act, cache_host_dev, \
            local_train_op = model
    
    # init session
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True), 
        allow_soft_placement=True, 
        log_device_placement=True))
    sess.run(init_op)
    
    # load data
    data = load_data(args)
    if args.party == "guest":
        train_input, train_labels, valid_input, valid_labels = data
    else:
        train_input, valid_input = data
    num_train = train_input.shape[0]
    num_valid = valid_input.shape[0]
    num_batch_per_workset = args.num_batch_per_workset
    num_update_per_batch = args.num_update_per_batch
    batch_size = args.batch_size
    num_train_batches = (num_train + batch_size - 1) // batch_size
    valid_batch_size = 100000
    num_valid_batches = (num_valid + valid_batch_size - 1) // valid_batch_size

    Federation.sync()
    logging.info("Start training...")
    train_start = time.time()

    class Cache(object):
        def __init__(self):
            self._cache = {}
            self._cv = threading.Condition()
        
        def put(self, batch, act, dev, timestamp):
            with self._cv:
                self._cache[batch] = [act, dev, timestamp, 0]
                self._cv.notify_all()
        
        def sample(self, reject_lists):
            while True:
                with self._cv:
                    while len(self._cache) == 0:
                        self._cv.wait()
                    ret = random.sample(self._cache.items(), 1)[0]
                    if ret[0] not in reject_lists:
                        return ret
        
        def inc(self, batch):
            with self._cv:
                if batch in self._cache:
                    self._cache[batch][-1] += 1
        
        def remove(self, batch):
            with self._cv:
                if batch in self._cache:
                    del self._cache[batch]

    num_total_comms = 0
    num_local_updates = 0
    is_finished = False
    cache = Cache()
    lock = threading.Lock()

    def local_update_fn():
        global num_total_comms, num_local_updates
        num_local_updates = 0
        max_staleness = num_batch_per_workset * num_update_per_batch
        try:
            prev_batches = []
            while not is_finished:
                # avoid repeatively stepping on a few batches
                batch, val = cache.sample(prev_batches)
                batch_cached_act, batch_cached_dev, \
                    batch_cached_at, batch_num_update \
                        = val
                if num_total_comms + num_local_updates - batch_cached_at >= max_staleness:
                    cache.remove(batch)
                    continue
                
                lock.acquire()
                batch_start = time.time()
                start_id = batch * batch_size
                end_id = start_id + batch_size

                if args.party == "guest":
                    batch_input = train_input[start_id : end_id]
                    batch_labels = train_labels[start_id : end_id].reshape(-1, 1)
                    loss_val, acc_val, logits_val, _ = sess.run(
                        [local_top_model["loss"], local_top_model["acc"], 
                            local_top_model["logits"], local_train_op], 
                        feed_dict={
                            guest_input: batch_input, 
                            labels: batch_labels, 
                            cache_host_act: batch_cached_act, 
                            cache_host_dev: batch_cached_dev, 
                        })
                else:
                    batch_input = train_input[start_id : end_id]
                    sess.run(
                        local_train_op, 
                        feed_dict={
                            host_input: batch_input, 
                            cache_host_act: batch_cached_act, 
                            cache_host_dev: batch_cached_dev, 
                        })
                
                batch_cost = time.time() - batch_start
                lock.release()

                if batch_num_update + 1 >= num_update_per_batch:
                    cache.remove(batch)
                else:
                    cache.inc(batch)
                prev_batches.append(batch)
                prev_batches = prev_batches[-(num_batch_per_workset - 1):]
                num_local_updates += 1
        except Exception as err:
            logging.error(f"Error in local update thread: {err}")
            import _thread
            _thread.interrupt_main() # raise a KeyboardInterrupt in main thread

    local_thread = threading.Thread(target=local_update_fn, daemon=True)
    local_thread.start()

    def valid_fn(num_updates):
        lock.acquire()
        logging.info(
            f"Validation after {num_total_comms + num_local_updates} updates "
            f"({num_total_comms} comm {num_local_updates} local)...")
        valid_start = time.time()
        if args.party == "guest":
            all_logits, all_labels = [], []
            batch_loss_list, batch_acc_list = [], []
        for batch in range(num_valid_batches):
            start_id = batch * valid_batch_size
            end_id = start_id + valid_batch_size

            if args.party == "guest":
                batch_input = valid_input[start_id : end_id]
                batch_labels = valid_labels[start_id : end_id].reshape(-1, 1)
                loss_val, acc_val, logits_val = sess.run(
                    [top_model["loss"], top_model["acc"], pred_op], 
                    feed_dict={
                        guest_input: batch_input, 
                        labels: batch_labels, 
                    })
                all_logits.append(logits_val.reshape(-1))
                all_labels.append(batch_labels.reshape(-1))
                batch_loss_list.append(loss_val)
                batch_acc_list.append(acc_val)
            else:
                batch_input = valid_input[start_id : end_id]
                sess.run(pred_op, feed_dict={host_input: batch_input})
        
        if args.party == "guest":
            loss = np.average(batch_loss_list)
            acc = np.average(batch_acc_list)
            all_logits = np.concatenate(all_logits)
            all_labels = np.concatenate(all_labels)
            fpr, tpr, _ = metrics.roc_curve(all_labels, all_logits)
            auc = metrics.auc(fpr, tpr)
            Federation.sync()
            logging.info(
                f"Validation after {num_total_comms + num_local_updates} updates "
                f"({num_total_comms} comm {num_local_updates} local) "
                f"cost {time.time() - valid_start:.4f} seconds, "
                f"{time.time() - train_start:.4f} elapsed, "
                f"Loss[{loss:.4f}] Accuracy[{acc:.4f}] AUC[{auc:.4f}]")
        else:
            Federation.sync()
            logging.info(
                f"Validation after {num_total_comms + num_local_updates} updates "
                f"({num_total_comms} comm {num_local_updates} local) "
                f"cost {time.time() - valid_start:.4f} seconds, "
                f"{time.time() - train_start:.4f} elapsed")
        lock.release()


    for epoch in range(args.epochs):
        for batch in range(num_train_batches):
            batch_start = time.time()
            start_id = batch * batch_size
            end_id = start_id + batch_size

            if args.party == "guest":
                batch_input = train_input[start_id : end_id]
                batch_labels = train_labels[start_id : end_id].reshape(-1, 1)
                loss_val, acc_val, logits_val, act_val, dev_val, _ = sess.run(
                    [top_model["loss"], top_model["acc"], top_model["logits"], 
                        host_act, host_dev, train_op], 
                    feed_dict={
                        guest_input: batch_input, 
                        labels: batch_labels, 
                    })
            else:
                batch_input = train_input[start_id : end_id]
                act_val, dev_val, _ = sess.run(
                    [host_act, host_dev, train_op], 
                    feed_dict={host_input: batch_input})
            cache.put(batch, act_val, dev_val, num_total_comms + num_local_updates)

            batch_cost = time.time() - batch_start
            num_total_comms += 1
            if num_total_comms % args.print_freq == 0:
                if args.party == "guest":
                    fpr, tpr, _ = metrics.roc_curve(batch_labels.reshape(-1), logits_val)
                    auc = metrics.auc(fpr, tpr)
                    logging.info(
                        f"Epoch[{epoch + 1}] Train "
                        f"Comm[{num_total_comms}] "
                        f"Local[{num_local_updates}] "
                        f"Time[{batch_cost:.4f}] "
                        f"Loss[{loss_val:.4f}] "
                        f"Accuracy[{acc_val:.4f}] "
                        f"AUC[{auc:.4f}]")
                else:
                    logging.info(
                        f"Epoch[{epoch + 1}] Train "
                        f"Comm[{num_total_comms}] "
                        f"Local[{num_local_updates}] "
                        f"Time[{batch_cost:.4f}]")
            
            if num_total_comms % args.valid_freq == 0:
                valid_fn(num_total_comms)
        
    is_finished = True
    local_thread.join()
    valid_fn(num_total_comms)
    logging.info("Training done")


def vanilla_vfl_main(args):
    # define model
    model = define_model(args)
    if args.party == "guest":
        guest_input, labels, top_model, train_op, pred_op = model
    else:
        host_input, train_op, pred_op = model
    
    # init session
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True), 
        allow_soft_placement=True, 
        log_device_placement=True))
    sess.run(init_op)
    
    # load data
    data = load_data(args)
    if args.party == "guest":
        train_input, train_labels, valid_input, valid_labels = data
    else:
        train_input, valid_input = data
    num_train = train_input.shape[0]
    num_valid = valid_input.shape[0]
    batch_size = args.batch_size
    num_train_batches = (num_train + batch_size - 1) // batch_size
    valid_batch_size = 100000
    num_valid_batches = (num_valid + valid_batch_size - 1) // valid_batch_size

    Federation.sync()
    logging.info("Start training...")
    train_start = time.time()

    def valid_fn(num_updates):
        logging.info(f"Validation after {num_updates} updates...")
        valid_start = time.time()
        if args.party == "guest":
            all_logits, all_labels = [], []
            batch_loss_list, batch_acc_list = [], []
        for batch in range(num_valid_batches):
            start_id = batch * valid_batch_size
            end_id = start_id + valid_batch_size

            if args.party == "guest":
                batch_input = valid_input[start_id : end_id]
                batch_labels = valid_labels[start_id : end_id].reshape(-1, 1)
                loss_val, acc_val, logits_val = sess.run(
                    [top_model["loss"], top_model["acc"], pred_op], 
                    feed_dict={
                        guest_input: batch_input, 
                        labels: batch_labels, 
                    })
                all_logits.append(logits_val.reshape(-1))
                all_labels.append(batch_labels.reshape(-1))
                batch_loss_list.append(loss_val)
                batch_acc_list.append(acc_val)
            else:
                batch_input = valid_input[start_id : end_id]
                sess.run(pred_op, feed_dict={host_input: batch_input})
        
        if args.party == "guest":
            loss = np.average(batch_loss_list)
            acc = np.average(batch_acc_list)
            all_logits = np.concatenate(all_logits)
            all_labels = np.concatenate(all_labels)
            fpr, tpr, _ = metrics.roc_curve(all_labels, all_logits)
            auc = metrics.auc(fpr, tpr)
            Federation.sync()
            logging.info(
                f"Validation after {num_updates} updates "
                f"cost {time.time() - valid_start:.4f} seconds, "
                f"{time.time() - train_start:.4f} elapsed, "
                f"Loss[{loss:.4f}] Accuracy[{acc:.4f}] AUC[{auc:.4f}]")
        else:
            Federation.sync()
            logging.info(
                f"Validation after {num_updates} updates "
                f"cost {time.time() - valid_start:.4f} seconds, "
                f"{time.time() - train_start:.4f} elapsed")

    num_total_comms = 0
    for epoch in range(args.epochs):
        for batch in range(num_train_batches):
            batch_start = time.time()
            start_id = batch * batch_size
            end_id = start_id + batch_size

            if args.party == "guest":
                batch_input = train_input[start_id : end_id]
                batch_labels = train_labels[start_id : end_id].reshape(-1, 1)
                loss_val, acc_val, logits_val, _ = sess.run(
                    [top_model["loss"], top_model["acc"], top_model["logits"], train_op], 
                    feed_dict={
                        guest_input: batch_input, 
                        labels: batch_labels, 
                    })
            else:
                batch_input = train_input[start_id : end_id]
                sess.run(train_op, feed_dict={host_input: batch_input})
            
            batch_cost = time.time() - batch_start
            num_total_comms += 1
            if num_total_comms % args.print_freq == 0:
                if args.party == "guest":
                    fpr, tpr, _ = metrics.roc_curve(batch_labels.reshape(-1), logits_val)
                    auc = metrics.auc(fpr, tpr)
                    logging.info(
                        f"Epoch[{epoch + 1}] Train "
                        f"Comm[{num_total_comms}] "
                        f"Time[{batch_cost:.4f}] "
                        f"Loss[{loss_val:.4f}] "
                        f"Accuracy[{acc_val:.4f}] "
                        f"AUC[{auc:.4f}]")
                else:
                    logging.info(
                        f"Epoch[{epoch + 1}] Train "
                        f"Comm[{num_total_comms}] "
                        f"Time[{batch_cost:.4f}]")
            
            if num_total_comms % args.valid_freq == 0:
                valid_fn(num_total_comms)

    if num_total_comms % args.valid_freq != 0:
        valid_fn(num_total_comms)
    logging.info("Training done")


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    args = parse_args()
    Federation.init_federation(args.task_id, args.party, args.config_file)
    if args.num_update_per_batch == 1:
        vanilla_vfl_main(args)
    else:
        assert args.num_batch_per_workset > 1
        celu_vfl_main(args)
    Federation.shutdown_federation()
