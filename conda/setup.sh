#!/usr/bin/env bash
set -euo pipefail

# ========= Config =========
ENV_NAME="cosmo_env"
PYTHON_VER="3.10"
CUDA_VER="12.1"        # e.g., 12.1 or 12.4; set USE_CPU=true for CPU-only
USE_CPU=false
REQ_FILE="requirements.txt"
# =========================

# ---- Find the conda binary (no interactive init required) ----
CONDA_BIN="${CONDA_EXE:-}"
if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
  elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
  elif [[ -x "/opt/conda/bin/conda" ]]; then
    CONDA_BIN="/opt/conda/bin/conda"
  else
    echo "[ERROR] Could not locate the 'conda' executable. Make sure Conda is installed."
    exit 1
  fi
fi

# Base dir for optional sourcing (not required for using $CONDA_BIN subcommands)
BASE_DIR="$("$CONDA_BIN" info --base)"
if [[ -f "$BASE_DIR/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "$BASE_DIR/etc/profile.d/conda.sh" || true
fi

# ---- Ensure environment exists ----
echo "==> Create (or ensure) environment: $ENV_NAME (Python $PYTHON_VER)"
if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "    Environment already exists, skipping creation."
else
  "$CONDA_BIN" create -y -n "$ENV_NAME" "python=$PYTHON_VER"
fi

# ---- Choose a mamba command ----
MAMBA_CMD="mamba"
if ! command -v mamba >/dev/null 2>&1; then
  echo "==> Installing mamba into base (from conda-forge)"
  "$CONDA_BIN" install -y -n base -c conda-forge mamba
  MAMBA_CMD="$CONDA_BIN run -n base mamba"
fi

# (Optional) Prefer apt mamba — uncomment if desired
# if command -v apt >/dev/null 2>&1; then
#   echo "==> Installing mamba via apt (Ubuntu/Debian)"
#   sudo apt update && sudo apt install -y mamba
#   MAMBA_CMD="mamba"
# fi

# ---- Install PyTorch ----
echo "==> Install PyTorch and CUDA/CPU packages into $ENV_NAME"
if [[ "$USE_CPU" == true ]]; then
  # CPU build
  $MAMBA_CMD install -y -n "$ENV_NAME" -c pytorch pytorch torchvision cpuonly
else
  # CUDA build
  $MAMBA_CMD install -y -n "$ENV_NAME" -c pytorch -c nvidia pytorch torchvision "pytorch-cuda=$CUDA_VER"
fi

# ---- Requirements ----
echo "==> Install requirements.txt (if present)"
if [[ -f "$REQ_FILE" ]]; then
  $MAMBA_CMD run -n "$ENV_NAME" pip install -r "$REQ_FILE"
else
  echo "    $REQ_FILE not found, skipping."
fi

# ---- Common packages ----
echo "==> Install common packages (numpy, matplotlib, tqdm, scikit-learn, ipykernel)"
$MAMBA_CMD run -n "$ENV_NAME" pip install -U numpy matplotlib tqdm scikit-learn ipykernel

# ---- Jupyter kernel ----
echo "==> Register Jupyter kernel: Python ($ENV_NAME)"
$MAMBA_CMD run -n "$ENV_NAME" python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo "==> All done ✅"
echo
echo "Usage:"
echo "  conda activate $ENV_NAME"
echo "  python -c 'import torch, sklearn, matplotlib, numpy; print(\"OK\")'"

