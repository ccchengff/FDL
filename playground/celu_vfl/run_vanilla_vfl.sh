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

cmd="python3 -u main.py \
    --party=${party} \
    --data=${data} \
    --num-update-per-batch=1"

echo "Running command: $cmd"
$cmd
