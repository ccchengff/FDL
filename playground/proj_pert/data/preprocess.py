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


def preprocess_movielens_1m():
    # source: https://grouplens.org/datasets/movielens/1m/
    if not os.path.exists("./movielens_1m/ml-1m.zip"):
        url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        command(f"wget --no-check-certificate {url} -P ./movielens_1m/")
    command(f"unzip ./movielens_1m/ml-1m.zip -d ./movielens_1m/")

    def load_dat(filename):
        ret = []
        with open(filename, encoding="ISO-8859-1") as fd:
            for line in fd:
                ret.append(line.strip().split("::"))
        return np.array(ret)

    # process movie data
    movies = pd.read_csv("./movielens_1m/ml-1m/movies.dat", sep="\:\:", header=None)
    movies.columns = ["MovieID", "Title", "Genres"]
    movies.set_index("MovieID", inplace=True)
    movies["Year"] = movies["Title"].apply(lambda x: x[-6:])
    movies["Title"] = movies["Title"].apply(lambda x: x[:-7])
    movies = process_sparse_feats(movies, ["Title", "Genres", "Year"])

    # process user data
    users = pd.read_csv("./movielens_1m/ml-1m/users.dat", sep="\:\:", header=None)
    users.columns = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
    users.set_index("UserID", inplace=True)
    users = process_sparse_feats(users, ["Gender", "Age", "Occupation", "Zip-code"])

    # rating data
    ratings = pd.read_csv("./movielens_1m/ml-1m/ratings.dat", sep="\:\:", header=None)
    ratings.columns = ["UserID", "MovieID", "Rating", "Timestamp"]
    ratings["Watched"] = 1.0
    ratings.drop(["Rating", "Timestamp"], axis=1, inplace=True)

    # negative sampling
    np.random.seed(1234)
    movie_ids = movies.index.tolist()
    neg_rate = 20
    neg_df = []
    for user_id, user_clicked_df in ratings.groupby("UserID"):
        user_watched = set(user_clicked_df["MovieID"].values.tolist())
        num_user_neg = len(user_watched) * neg_rate
        cnt = 0
        while cnt < num_user_neg:
            movie_id = movie_ids[np.random.randint(0, len(movie_ids))]
            if movie_id not in user_watched:
                neg_df.append([user_id, movie_id])
                cnt += 1

    neg_df = pd.DataFrame(neg_df, columns=["UserID", "MovieID"])
    neg_df["Watched"] = 0.0
    df = pd.concat([ratings, neg_df])

    # join features
    movies_feats = pd.merge(df, movies, on="MovieID", how="left")[["Title", "Genres", "Year"]]
    users_feats = pd.merge(df, users, on="UserID", how="left")[["Gender", "Age", "Occupation", "Zip-code"]]
    labels = df["Watched"]

    # split into train and valid sets
    num_data = labels.shape[0]
    num_valid = num_data // 10
    perm = np.random.permutation(num_data)
    train_movies = movies_feats.iloc[perm[:-num_valid]].astype(np.int32)
    valid_movies = movies_feats.iloc[perm[-num_valid:]].astype(np.int32)
    train_users = users_feats.iloc[perm[:-num_valid]].astype(np.int32)
    valid_users = users_feats.iloc[perm[-num_valid:]].astype(np.int32)
    train_labels = labels.iloc[perm[:-num_valid]].astype(np.float32)
    valid_labels = labels.iloc[perm[-num_valid:]].astype(np.float32)

    # save data
    logging.info("Saving data...")
    logging.info(
        f"Train size: guest[{train_movies.shape}] " + 
        f"host[{train_users.shape}] " + 
        f"labels[{train_labels.shape}]")
    np.savez("./movielens_1m/train.npz",
             guest=train_movies,
             host=train_users,
             labels=train_labels)
    logging.info(
        f"Valid size: guest[{valid_movies.shape}] " + 
        f"host[{valid_users.shape}] " + 
        f"labels[{valid_labels.shape}]")
    np.savez("./movielens_1m/valid.npz",
             guest=valid_movies,
             host=valid_users,
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


def command(cmd):
    logging.info(f">>> Executing command \"{cmd}\"...")
    ret = os.system(cmd)
    assert ret == 0, f"Command failed, return code: {ret}"
    logging.info(">>> Command done")


if __name__ == "__main__":
    try:
        dataset = sys.argv[1]
    except:
        logging.error("Missing dataset (criteo|avazu|movielens_1m)")
        sys.exit(1)

    logging.info(f"Preprocssing {dataset}")
    if dataset == "criteo":
        preprocess_criteo()
    elif dataset == "avazu":
        preprocess_avazu()
    elif dataset == "movielens_1m":
        preprocess_movielens_1m()
    else:
        raise ValueError(f"No such dataset: {dataset}")
