#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${AMP_STATS_PYTHON:-/home/omics/jinzy/software/envs/amp_esmc_tenfold_v1/bin/python}"
STAGE="${1:-check}"
B="${BOOTSTRAP:-10000}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
cd "$ROOT"
test -x "$PY" || { echo "Set AMP_STATS_PYTHON to a Python >=3.10 environment with requirements.txt" >&2; exit 2; }
case "$STAGE" in
 check)
  "$PY" -B code/verify_package.py
  "$PY" -B code/test_analysis.py
  ;;
 stats)
  test "$B" -ge 10000 || { echo "Production stats requires BOOTSTRAP>=10000; use smoke for short tests" >&2; exit 2; }
  "$PY" -B code/verify_package.py
  mkdir -p results
  "$PY" -B code/selective.py --bootstrap "$B" --output results/loss
  "$PY" -B code/transfer.py --bootstrap "$B" --output results/transfer
  "$PY" -B code/plot_results.py --loss results/loss --transfer results/transfer --output figures
  ;;
 screen)
  "$PY" -B code/verify_package.py
  "$PY" -B code/screen_panel.py --output results/panel_screen
  ;;
 panel)
  "$PY" -B code/verify_package.py
  test "$B" -ge 10000 || { echo "Production panel analysis requires BOOTSTRAP>=10000" >&2; exit 2; }
  "$PY" -B code/infer_panel.py check --output results/panel_inference
  "$PY" -B code/infer_panel.py run --output results/panel_inference
  "$PY" -B code/panel_metrics.py --bootstrap "$B" --inference results/panel_inference --screen results/panel_screen --output results/panel_metrics
  "$PY" -B code/plot_results.py --loss results/loss --transfer results/transfer --panel results/panel_metrics --output figures
  ;;
 smoke)
  "$PY" -B code/verify_package.py
  "$PY" -B code/test_analysis.py
  "$PY" -B code/selective.py --bootstrap 100 --output QA/loss_smoke
  "$PY" -B code/transfer.py --bootstrap 40 --output QA/transfer_smoke
  "$PY" -B code/plot_results.py --loss QA/loss_smoke --transfer QA/transfer_smoke --output QA/figure_smoke --allow-smoke-preview
  ;;
 *) echo "Usage: bash run_all.sh {check|stats|screen|panel|smoke}" >&2; exit 2 ;;
esac
