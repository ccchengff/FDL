# CELU-VFL

This is a lite version of codes for our paper entitled *Towards Communication-efficient Vertical Federated Learning Training via Cache-enabled Local Updates*.

## Environment
We conduct all experiments on two servers with Ubuntu 18.04 and Python 3.6.9. Please clone this repo and execute the following commands under the `FDL` directory:
```shell
# install the dependencies
$ python3 -m pip install -r requirements.txt
# compile the protobuf
$ python3 -m grpc_tools.protoc \
    --proto_path=./protos/ \
    --python_out=./python/ \
    --grpc_python_out=./python/ \
    $(find ./protos -name "*.proto")
```

The experiments should be run on *two servers connected with WAN*. Please configure your server IPs in `config.yaml`. 
*Note: You may also simulate the experiments on one server (e.g., setting the IPs as localhost), however, the effectiveness of our work will not be significant since the cross-party communication will no longer be the bottleneck.*

## Dataset Preparation
The datasets are expected to be downloaded manually in advance. Please download and unzip the [Criteo](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) and [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction/data) datasets to the `data/criteo` and `data/avazu` directories, respectively. Next, you may preprocess the datasets via
```shell
$ cd data; python3 preprocess.py [criteo|avazu] # it may take a while
```

## Experiments
We prepare scripts for our experiments. For instance, you can run the CELU-VFL training on two servers accordingly:
```shell
# vim run_celu_vfl.sh # edit the hyper-parameters or dataset if needed
$ CUDA_VISIBLE_DEVICES=0 bash run_celu_vfl.sh host # to lauch Party A
$ CUDA_VISIBLE_DEVICES=0 bash run_celu_vfl.sh guest # to lauch Party B
```

