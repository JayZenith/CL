#!/usr/bin/env bash
# Creates .venv with prebuilt wheels only. Picks the torch CUDA build the driver supports.
set -euo pipefail
cd "$(dirname "$0")"

TORCH=2.6.0
PYTHON=3.12

driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
at_least() { [ "$(printf '%s\n%s\n' "$1" "$driver" | sort -V | head -n1)" = "$1" ]; }
if at_least 550.54; then cuda=cu124      # CUDA 12.4
elif at_least 520.61; then cuda=cu118    # CUDA 11.8
else echo "Driver $driver is too old: need >= 520.61" >&2; exit 1
fi
echo "Driver $driver -> torch $TORCH+$cuda"

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv venv --python "$PYTHON" --allow-existing .venv
uv pip install --python .venv/bin/python --only-binary :all: "torch==$TORCH" --index-url "https://download.pytorch.org/whl/$cuda"
uv pip install --python .venv/bin/python --only-binary :all: -r requirements.txt

.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.is_bf16_supported(), "BF16 unsupported"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
