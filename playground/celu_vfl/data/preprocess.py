import os, sys, traceback
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import logging
logging.basicConfig(format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

def preprocess_avazu():
    # source: https://www.kaggle.com/c/avazu-ctr-prediction/data
    data_path = "./avazu/train"
    assert os.path.exists(data_path), \
        "Please download train.gz and extract to ./avazu/ in advance"
    
    # load data
    logging.info("Reading data...")
    df = pd.read_csv(data_path)

    # preprocess data
    logging.info("Preprocessing data...")
    labels = df["click"]
    feats = process_sparse_feats(df, df.columns[2:])
    guest_feats = feats[df.columns[-8:]] # C14-C21
    host_feats = feats[df.columns[2:-8]]

    # split into train and valid sets
    num_data = feats.shape[0]
    num_valid = num_data // 10
    perm = np.random.permutation(num_data)
    train_guest = guest_feats.iloc[perm[:-num_valid]].astype(np.int32)
    valid_guest = guest_feats.iloc[perm[-num_valid:]].astype(np.int32)
    train_host = host_feats.iloc[perm[:-num_valid]].astype(np.int32)
    valid_host = host_feats.iloc[perm[-num_valid:]].astype(np.int32)
    train_labels = labels.iloc[perm[:-num_valid]].astype(np.float32)
    valid_labels = labels.iloc[perm[-num_valid:]].astype(np.float32)

    # save data
    logging.info("Saving data...")
    logging.info(
        f"Train size: guest[{train_guest.shape}] " + 
        f"host[{train_host.shape}] " + 
        f"labels[{train_labels.shape}]")
    np.savez("./avazu/train.npz",
             guest=train_guest,
             host=train_host,
             labels=train_labels)
    logging.info(
        f"Valid size: guest[{valid_guest.shape}] " + 
        f"host[{valid_host.shape}] " + 
        f"labels[{valid_labels.shape}]")
    np.savez("./avazu/valid.npz",
             guest=valid_guest,
             host=valid_host,
             labels=valid_labels)

    logging.info("Data preprocessing done")

def preprocess_criteo():
    # source: http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/
    data_path = "./criteo/train.txt"
    assert os.path.exists(data_path), \
        "Please download dac.tar.gz and extract to ./criteo/ in advance"

    # load data
    logging.info("Reading data...")
    df = pd.read_csv(data_path, sep='\t', header=None)
    df.columns = ["labels"] + ["I%d"%i for i in range(1,14)] + ["C%d"%i for i in range(14,40)]

    # preprocess dense and sparse data
    logging.info("Preprocessing data...")
    labels = df["labels"]
    dense_feats =  [col for col in df.columns if col.startswith('I')]
    sparse_feats = [col for col in df.columns if col.startswith('C')]
    dense_feats = process_dense_feats(df, dense_feats)
    sparse_feats = process_sparse_feats(df, sparse_feats)

    # split into train and valid sets
    num_data = dense_feats.shape[0]
    num_valid = num_data // 10
    perm = np.random.permutation(num_data)
    train_dense = dense_feats.iloc[perm[:-num_valid]].astype(np.float32)
    valid_dense = dense_feats.iloc[perm[-num_valid:]].astype(np.float32)
    train_sparse = sparse_feats.iloc[perm[:-num_valid]].astype(np.int32)
    valid_sparse = sparse_feats.iloc[perm[-num_valid:]].astype(np.int32)
    train_labels = labels.iloc[perm[:-num_valid]].astype(np.float32)
    valid_labels = labels.iloc[perm[-num_valid:]].astype(np.float32)

    # save data
    logging.info("Saving data...")
    logging.info(
        f"Train size: guest[{train_dense.shape}] " + 
        f"host[{train_sparse.shape}] " + 
        f"labels[{train_labels.shape}]")
    np.savez("./criteo/train.npz",
             guest=train_dense,
             host=train_sparse,
             labels=train_labels)
    logging.info(
        f"Valid size: guest[{valid_dense.shape}] " + 
        f"host[{valid_sparse.shape}] " + 
        f"labels[{valid_labels.shape}]")
    np.savez("./criteo/valid.npz",
             guest=valid_dense,
             host=valid_sparse,
             labels=valid_labels)

    logging.info("Data preprocessing done")


def process_dense_feats(data, feats):
    logging.info(f"Processing feats: {feats}")
    d = data.copy()
    d = d[feats].fillna(0.0)
    for f in feats:
        d[f] = d[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)
    return d


def process_sparse_feats(data, feats):
    logging.info(f"Processing feats: {feats}")
    d = data.copy()
    d = d[feats].fillna("-1")
    for f in feats:
        label_encoder = LabelEncoder()
        d[f] = label_encoder.fit_transform(d[f])
    feature_cnt = 0
    for f in feats:
        d[f] += feature_cnt
        feature_cnt += d[f].nunique()
    return d

if __name__ == "__main__":
    try:
        dataset = sys.argv[1]
    except:
        logging.error("Missing dataset (criteo|avazu)")
        sys.exit(1)

    logging.info(f"Preprocssing {dataset}")
    if dataset == "criteo":
        preprocess_criteo()
    elif dataset == "avazu":
        preprocess_avazu()
    else:
        raise ValueError(f"No such dataset: {dataset}")
