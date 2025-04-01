#!/bin/bash
REPO_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ -z $1 ]; then
    exit
fi

rm -rf ${REPO_DIR}/ckpts/$1

#
#
MADRONA_MWGPU_FORCE_DEBUG=1 MAD_TRADE_DEBUG_WAIT=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false MADRONA_LEARN_DUMP_LOWERED=/tmp/lowered MADRONA_LEARN_DUMP_IR=/tmp/ir MADRONA_MWGPU_KERNEL_CACHE="${REPO_DIR}/build/cache" \
  python "${REPO_DIR}/scripts/jax_train.py" \
    --ckpt-dir ${REPO_DIR}/ckpts/ \
    --tb-dir "${REPO_DIR}/tb" \
    --run-name $1 \
    --num-updates 1000000 \
    --num-worlds 1 \
    --num-agents 4 \
    --num-npcs 1 \
    --settlement-price 100 \
    --lr 1e-4 \
    --steps-per-update 40 \
    --num-bptt-chunks 2 \
    --num-minibatches 4 \
    --entropy-loss-coef 0.01 \
    --value-loss-coef 1.0 \
    --pbt-ensemble-size 1 \
    --pbt-past-policies 0 \
    --profile-port 5000 \
    --bf16 \
    --eval-frequency 500 \
    --gpu-sim
