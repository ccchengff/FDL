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
    parser.add_argument("--task-id", type=str, default="test_task", 
                        help="Task id")
    parser.add_argument("--party", type=str, default="guest", 
                        choices=("guest", "host"), help="Party name")
    parser.add_argument("--config-file", type=str, default="config.yaml", 
                        help="Config yaml")
    parser.add_argument("--data", type=str, 
                        choices=("criteo", "avazu", "movielens_1m"), 
                        help="Name of dataset")
    parser.add_argument("--device", type=str, default="/gpu:0", 
                        help="Name of device")
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Max number of epochs")
    parser.add_argument("--eta", type=float, default=0.1, 
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1024, 
                        help="Batch size")
    parser.add_argument("--valid-freq", type=int, default=5000, 
                        help="Frequency of evaluation on valid set")
    parser.add_argument("--print-freq", type=int, default=20, 
                        help="Frequency of printing metrics")
    parser.add_argument("--perturb", type=str, default="iso-proj", 
                        choices=("proj", "iso-proj", "marvell", "gaussian", "maxnorm_gaussian", 
                                 "laplace_dp", "bernouli_dp", "mixpro", "none"), 
                        help="Perturbation method")
    parser.add_argument("--sum-kl-bound", type=float, default=4.0, 
                        help="Upper bound of sum KL divergence (for proj and iso-proj")
    parser.add_argument("--init-scale", type=float, default=4.0, 
                        help="Initial value of P is scale * g (for marvell)")
    parser.add_argument("--dp-eps", type=float, default=1.0, 
                        help="Epsilon for differential privacy")
    parser.add_argument("--mixpro-alpha", type=float, default=0.6, 
                        help="Parameter in Beta Distribution for MixPro")
    parser.add_argument("--mixpro-cos-sim-thres", type=float, default=(0.5 * np.sqrt(3)), 
                        help="Threshold of Cosine Similarity for MixPro")

    args = parser.parse_args()
    logging.info(f"Args: {args}")
    return args


def main(args):
    # define model
    model = define_model(args)
    if args.party == "guest":
        guest_input, labels, top_model, train_op, pred_op = model
    else:
        host_input, train_op, pred_op, infer_logits = model
        infer_names = [t[0] for t in infer_logits]
        infer_fetches = [t[1] for t in infer_logits]

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
        # Here we let host load labels for 
        # the evaluation of label inference methods
        train_input, train_labels, valid_input, valid_labels = data
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

    num_total_updates = 0
    for epoch in range(args.epochs):
        for batch in range(num_train_batches):
            batch_start = time.time()
            start_id = batch * batch_size
            end_id = start_id + batch_size
            need_print = ((num_total_updates + 1) % args.print_freq == 0)

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
                if need_print:
                    infer_vals = sess.run(
                        [train_op, *infer_fetches], 
                        feed_dict={host_input: batch_input})[1:]
                else:
                    sess.run(train_op, feed_dict={host_input: batch_input})
            
            batch_cost = time.time() - batch_start
            num_total_updates += 1
            if need_print:
                if args.party == "guest":
                    batch_labels = batch_labels.reshape(-1)
                    fpr, tpr, _ = metrics.roc_curve(batch_labels, logits_val)
                    auc = metrics.auc(fpr, tpr)
                    logging.info(
                        f"Train Update[{num_total_updates}] "
                        f"Time[{batch_cost:.4f}] "
                        f"Loss[{loss_val:.4f}] "
                        f"Accuracy[{acc_val:.4f}] "
                        f"AUC[{auc:.4f}]")
                else:
                    batch_labels = train_labels[start_id : end_id].reshape(-1)
                    infer_aucs = []
                    for infer_val in infer_vals:
                        fpr, tpr, _ = metrics.roc_curve(batch_labels, infer_val)
                        infer_auc = metrics.auc(fpr, tpr)
                        infer_auc = max(infer_auc, 1.0 - infer_auc)
                        infer_aucs.append(infer_auc)
                    infer_msg = " ".join(f"{t[0]}AUC[{t[1]:.4f}]" for t in zip(infer_names, infer_aucs))
                    logging.info(
                        f"Train Update[{num_total_updates}] "
                        f"Time[{batch_cost:.4f}] "
                        + infer_msg)
            
            if num_total_updates % args.valid_freq == 0:
                valid_fn(num_total_updates)

    if num_total_updates % args.valid_freq != 0:
        valid_fn(num_total_updates)
    logging.info("Training done")


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    args = parse_args()
    Federation.init_federation(args.task_id, args.party, args.config_file)
    main(args)
    Federation.shutdown_federation()
