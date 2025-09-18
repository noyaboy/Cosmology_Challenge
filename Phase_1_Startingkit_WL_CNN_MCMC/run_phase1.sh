#!/usr/bin/env bash
set -euo pipefail

# Run Phase 1 CNN+MCMC pipeline converted from notebook
# Usage: ./run_phase1.sh [python_args...]
# Example: ./run_phase1.sh --help

#     # Create venv if not exists
#     if [ ! -d ".venv" ]; then
#       python3 -m venv .venv
#     fi
#     
#     # Activate venv
#     source .venv/bin/activate
#     
#     # Upgrade pip
#     python -m pip install --upgrade pip
#     
#     # Install dependencies
#     python -m pip install numpy scipy matplotlib tqdm scikit-learn
#     # Install torch (CPU-only by default). For CUDA, adjust according to your environment.
#     python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || python -m pip install torch torchvision

# Run the converted script
MPLBACKEND=Agg python Phase_1_Startingkit_WL_CNN_MCMC.py "$@"
