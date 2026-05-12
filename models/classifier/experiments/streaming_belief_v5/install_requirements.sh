#!/usr/bin/env bash
set -euo pipefail

# Install PyTorch first so extension packages can import it during build.
python -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install the pure-Python/common dependencies separately from the CUDA extensions.
python -m pip install \
  "transformers==4.46.3" \
  "numpy>=1.24.0" \
  "scikit-learn>=1.3.0" \
  "peft>=0.11.0" \
  packaging \
  ninja \
  wheel

# These packages need torch available in the active environment at build time.
python -m pip install --no-build-isolation causal-conv1d==1.5.0.post8
python -m pip install --no-build-isolation mamba-ssm==2.2.4
