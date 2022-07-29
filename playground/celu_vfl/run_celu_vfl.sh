#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Missing arguments: <PARTY (guest|host)>"
    exit 1
fi

WORKSPACE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ${WORKSPACE}/../../env.exp
export PYTHONPATH="$PYTHONPATH:$WORKSPACE"
export DATASET_DIR="$WORKSPACE/data"

party=$1
data="criteo" # choices: criteo, avazu
W=5 # size of the workset table
R=5 # number of local updates
CosXi=0.5 # threshold of similarities

cmd="python3 -u main.py \
    --party=${party} \
    --data=${data} \
    --num-batch-per-workset=${W} \
    --num-update-per-batch=${R} \
    --sim-thres=${CosXi}"

echo "Running command: $cmd"
$cmd
