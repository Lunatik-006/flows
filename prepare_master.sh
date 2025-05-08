#!/usr/bin/env bash
set -e

# === 1.1  Обновляем систему и ставим драйвер + CUDA ===
sudo apt update && sudo apt -y upgrade
# Добавили git-lfs и libopenmpi-dev для Git Large File Storage и распределённого бекенда
sudo apt -y install nvidia-driver-535 cuda-drivers build-essential git rsync wget curl git-lfs libopenmpi-dev
# Перезагрузка драйвера требуется один раз
sudo reboot

# --------------------------------------------
# === 1.2  Устанавливаем Miniconda ===
cd /tmp
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p "$HOME/miniconda"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda init

# Создаём env только если его ещё нет
if ! conda env list | grep -q deepseek; then
  conda create -y -n deepseek python=3.10
fi
conda activate deepseek

# --------------------------------------------
# === 1.3  Ставим PyTorch + зависимости ===
# Переходим на установку через conda — стабильнее и сразу ставит совместимую версию CUDA 11.8
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install --upgrade pip
pip install transformers accelerate

# --------------------------------------------
# === 1.4  Скачиваем модель один раз на master ===
sudo mkdir -p /opt/models && cd /opt/models
git lfs install
if [ ! -d deepseek-coder-6.7b-base ]; then
  git clone https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base
fi
# ~13 ГБ в FP16; убедитесь, что хватит места на /opt
echo "[MASTER] Подготовка завершена. При необходимости перезагрузите машину."
