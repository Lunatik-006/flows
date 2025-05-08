#!/usr/bin/env bash
set -e

echo "[SLAVE] Шаг 1 — установка драйвера и базовых пакетов"

# Добавляем репозиторий NVIDIA (замените при необходимости версию Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"

sudo apt update && sudo apt -y upgrade
sudo apt -y install cuda-drivers-535 build-essential git rsync wget curl git-lfs libopenmpi-dev

echo "[SLAVE] Шаг 2 — установка Miniconda"
if [ ! -d "$HOME/miniconda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda"
fi
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda init
if ! conda env list | grep -q deepseek; then
  conda create -y -n deepseek python=3.10
fi
conda activate deepseek

echo "[SLAVE] Шаг 3 — установка PyTorch и зависимостей"
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install --upgrade pip
pip install transformers accelerate

echo "[SLAVE] Шаг 4 — копируем модель (если ещё не скопирована)"
if [ ! -d /opt/models/deepseek-coder-6.7b-base ]; then
  sudo mkdir -p /opt/models
  sudo rsync -az master:/opt/models/deepseek-coder-6.7b-base /opt/models/
  sudo chown -R $USER:$USER /opt/models
fi

echo "[SLAVE] Готов! Перезагрузка для активации драйвера..."

