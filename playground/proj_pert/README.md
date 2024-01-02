# ProjPert

This is a lite version of codes for our paper entitled *ProjPert: Projection-based Perturbation for Label Protection in Split Learning based Vertical Federated Learning*.

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

Optionally, you can compile the following library to speed up `ProjPert-opt` with OpenMP. If not, a sequential version will be invoked, which might be slower. Execute the following command under the `FDL` directory:
```shell
c++ -O3 -Wall -g -shared -std=c++11 -fPIC -fopenmp -w \
    `python3 -m pybind11 --includes` \
    python/fdl/binding/proj_pert_solver.cc \
    -o proj_pert_solver_c`python3-config --extension-suffix`
```

The experiments should be run on two servers connected with WAN/LAN. Please configure your server IPs in `config.yaml`. You may also simulate the experiments on one server (e.g., setting the IPs as localhost). 

## Dataset Preparation
We prepare the code for dataset preprocessing. The [Criteo](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) and [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction/data) datasets are expected to be downloaded manually in advance. Please download and unzip them to the `expr/data/criteo` and `expr/data/avazu` directories, respectively. Next, you may preprocess the datasets via
```shell
$ cd data; python3 preprocess.py [criteo|avazu|movielens_1m] # it may take a while
```

## Experiments
You can launch the training tasks using the `run.sh` script, which takes the following arguments:
- party: one of "host" (Party A) and "guest" (Party B).
- dataset: one of "criteo", "avazu", and "movielens_1m".
- perturb: one of "proj" (ProjPert-opt), "iso-proj" (ProjPert-iso), "marvell" (Marvell), and "none" (NoDefense).
- perturn param: for "proj" and "iso-proj", this is the sumKL threshold; for "marvell", this is the constraint of perturbation; for "none", this is optional.

For instance, you can train on the Criteo dataset with `ProjPert-iso`:
```shell
$ CUDA_VISIBLE_DEVICES=0 bash run.sh host criteo iso-proj 4.0  # lauch Party A on the first server
$ CUDA_VISIBLE_DEVICES=0 bash run.sh guest criteo iso-proj 4.0 # lauch Party B on the second server
```
Or with `Marvell`:
```shell
$ CUDA_VISIBLE_DEVICES=0 bash run.sh host criteo marvell 4.0  # lauch Party A on the first server
$ CUDA_VISIBLE_DEVICES=0 bash run.sh guest criteo marvell 4.0 # lauch Party B on the second server
```
Or without perturbation:
```shell
$ CUDA_VISIBLE_DEVICES=0 bash run.sh host criteo none  # lauch Party A on the first server
$ CUDA_VISIBLE_DEVICES=0 bash run.sh guest criteo none # lauch Party B on the second server
```
