#!/usr/bin/env bash
set -e
# -- 3.1 Параметры кластера ------------------------------------
MASTER_ADDR=192.168.31.160
MASTER_PORT=29500
HOSTS=($MASTER_ADDR 192.168.31.227 192.168.31.236 192.168.0.59)
NNODES=${#HOSTS[@]}
# --------------------------------------------------------------

start_node () {
  local node_ip=$1
  local node_rank=$2
  local cmd="
    source $HOME/miniconda/etc/profile.d/conda.sh &&     conda activate deepseek &&     cd /opt/models/deepseek-coder-6.7b-base &&     torchrun --nnodes=$NNODES --nproc_per_node=1              --node_rank=$node_rank              --master_addr=$MASTER_ADDR              --master_port=$MASTER_PORT              /opt/models/deepseek-coder-6.7b-base/inference.py
  "
  if [[ $node_ip == $MASTER_ADDR ]]; then
    echo "[MASTER] Стартую локальный процесс (rank=$node_rank)"
    eval "$cmd" &
  else
    echo "[MASTER] Стартую узел $node_ip (rank=$node_rank)"
    ssh "$node_ip" "$cmd" &
  fi
}

echo "[MASTER] Запускаем распределённый инференс DeepSeek-Coder-6.7B"
for idx in "${!HOSTS[@]}"; do
  start_node "${HOSTS[$idx]}" "$idx"
done
wait
