#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
tr -d '\r' < SHA256SUMS.txt | sha256sum -c -
PY="/home/omics/jinzy/software/envs/amp_esmc_tenfold_v1/bin/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
if [ "${1:-}" = "--status" ]; then
  exec "$PY" -B -u scheduler.py status
fi
if [ "${1:-}" != "--check-only" ] && [ "${1:-}" != "--confirm-resources" ]; then
  echo "用法：bash launch_server.sh --check-only | --confirm-resources | --status"
  exit 2
fi
"$PY" -B -u scheduler.py check
if [ "${1:-}" = "--check-only" ]; then exit 0; fi
SESSION="amp_tools_six_by_six_v1_2"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "会话已存在，不重复启动。查看：tmux attach -t $SESSION"
  exit 0
fi
tmux new-session -d -s "$SESSION" -c "$PWD" \
  'bash run_server.sh; TASK_STATUS=$?; printf "\nLauncher exit code: %s\n" "$TASK_STATUS"; exec bash'
echo "已启动：$SESSION"
echo "主日志：/home/omics/jinzy/ai-peptide/tools/published_amp_six_by_six_v1_2/scheduler_console.log"
echo "状态：bash launch_server.sh --status"
