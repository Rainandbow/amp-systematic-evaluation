#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
"/home/omics/jinzy/software/envs/amp_esmc_tenfold_v1/bin/python" -B -u scheduler.py run --confirm-resources \
  2>&1 | tee -a scheduler_console.log
