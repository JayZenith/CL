#!/usr/bin/env bash
# Builds .venv and fetches the paper's ImageNet classes and class order. Rerun to resume.
set -euo pipefail
cd "$(dirname "$0")"

COMMIT=3138c5556859e0e2918d40f113b40699b6575cfa
DATA=1-8pgLUKM1WBsjum_bw96PVb4veTZRykf

if [ ! -x .venv/bin/python ]; then
  # Prebuilt wheels only, with the torch CUDA build the driver supports.
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
  at_least() { [ "$(printf '%s\n%s\n' "$1" "$driver" | sort -V | head -n1)" = "$1" ]; }
  if at_least 550.54; then cuda=cu124
  elif at_least 520.61; then cuda=cu118
  else echo "Driver $driver is too old: need >= 520.61" >&2; exit 1
  fi
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
  uv venv --python 3.12 --allow-existing .venv
  uv pip install --python .venv/bin/python --only-binary :all: torch==2.6.0 --index-url "https://download.pytorch.org/whl/$cuda"
  uv pip install --python .venv/bin/python --only-binary :all: -r requirements.txt
fi
mkdir -p data

if [ ! -f data/classes/999.npy ]; then
  # Drive drops long downloads now and then; resume until done.
  until curl -L -C - --retry 5 -o data/classes.zip \
    "https://drive.usercontent.google.com/download?id=$DATA&export=download&confirm=t"; do sleep 5; done
  .venv/bin/python -m zipfile -e data/classes.zip data/
  rm data/classes.zip
fi

if [ ! -f data/class_order.npy ]; then
  curl -fL -o data/class_order.pkl \
    "https://github.com/shibhansh/loss-of-plasticity/raw/$COMMIT/lop/imagenet/class_order"
  .venv/bin/python - <<'PY'
import pickle
import numpy as np
with open("data/class_order.pkl", "rb") as f:
    np.save("data/class_order.npy", np.array(pickle.load(f)))
PY
  rm data/class_order.pkl
fi

.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
order = np.load("data/class_order.npy")
x = np.load("data/classes/0.npy")
n = len(list(Path("data/classes").glob("*.npy")))
assert n == 1000, n
print(f"{n} classes, each {x.shape} {x.dtype} range {x.min()}-{x.max()}; order {order.shape}")
PY
