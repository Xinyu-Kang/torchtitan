#!/usr/bin/bash
# Multi-node launcher with optional TorchComms integration.
# Designed for Slurm (uses SLURM_* envs when present) but also works with manual MASTER_ADDR/MASTER_PORT.

set -ex

# Basic job config
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/llama3_8b.toml"}

# TorchComms toggle. When disabled we fall back to the default torchtitan trainer.
USE_TORCHCOMMS=${USE_TORCHCOMMS:-"1"}
if [ "${USE_TORCHCOMMS}" = "1" ]; then
    TRAIN_FILE=${TRAIN_FILE:-"torchtitan.experiments.torchcomms.train"}
    TEST_BACKEND=${TEST_BACKEND:-"rcclx"}   # rccl/rcclx depending on cluster
else
    TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}
fi

# Rendezvous settings (prefer Slurm-derived values if present).
DEFAULT_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -n 1 || true)
MASTER_ADDR=${MASTER_ADDR:-${SLURM_MASTER_ADDR:-${DEFAULT_MASTER_ADDR:-"127.0.0.1"}}}
MASTER_PORT=${MASTER_PORT:-${SLURM_MASTER_PORT:-"29500"}}
NNODES=${NNODES:-${SLURM_NNODES:-"1"}}
NODE_RANK=${NODE_RANK:-${SLURM_PROCID:-${SLURM_NODEID:-"0"}}}

# Network / comm defaults (aligned with torchtitan-amd script).
NCCL_NET=${NCCL_NET:-"IB"}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"enp49s0f0np0"}
NCCL_IB_HCA=${NCCL_IB_HCA:-"bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8"}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-"0"}
NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-"3"}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://${MASTER_ADDR}:29510"}
TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-"false"}
NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}
NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-"ALL"}

extra_args=""
if [ $# -ne 0 ]; then
    extra_args="$*"
fi

env \
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    NCCL_NET=${NCCL_NET} \
    ${NCCL_SOCKET_IFNAME:+NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}} \
    ${NCCL_IB_HCA:+NCCL_IB_HCA=${NCCL_IB_HCA}} \
    ${NCCL_IB_GID_INDEX:+NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}} \
    ${NCCL_IB_DISABLE:+NCCL_IB_DISABLE=${NCCL_IB_DISABLE}} \
    NCCL_DEBUG=${NCCL_DEBUG} \
    NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS} \
    ${TEST_BACKEND:+TEST_BACKEND=${TEST_BACKEND}} \
    TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM} \
    torchrun --nnodes=${NNODES} \
             --node_rank=${NODE_RANK} \
             --nproc_per_node=${NGPU} \
             --rdzv_backend=c10d \
             --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
             --local-ranks-filter ${LOG_RANK} \
             --role rank --tee 3 \
             -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} ${extra_args}
