#!/usr/bin/env bash
# 第 3 步：观众 Agent 并行"看片"。
# 7B 观众: GPUS_PER_PROC=1（默认，8 个进程）
# 72B 观众: GPUS_PER_PROC=4（2 个进程，每个用 4 张卡），并在配置里把 viewer.model_id 改成 72B
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/experiment.yaml}
NUM_GPUS=${NUM_GPUS:-8}
GPUS_PER_PROC=${GPUS_PER_PROC:-1}
NPROC=$((NUM_GPUS / GPUS_PER_PROC))
mkdir -p logs
for ((r=0; r<NPROC; r++)); do
  start=$((r * GPUS_PER_PROC))
  devs=$(seq -s, $start $((start + GPUS_PER_PROC - 1)))
  CUDA_VISIBLE_DEVICES=$devs python -m src.run_viewer --config "$CONFIG" \
      --rank $r --world_size $NPROC "$@" > logs/viewer_rank$r.log 2>&1 &
  echo "启动 viewer rank $r (GPU $devs)，日志: logs/viewer_rank$r.log"
done
wait
echo "观众评估完成。"
