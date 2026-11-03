#!/usr/bin/env bash
# 第 1 步：8 卡并行逐镜头生成视频。每张卡一个进程，按 rank 分任务。
# 用法: bash scripts/run_generate.sh            （默认 8 卡）
#       NUM_GPUS=4 bash scripts/run_generate.sh
#       STAGGER=60 bash scripts/run_generate.sh  （每个进程间隔 60 秒启动，降低同时加载模型时的内存峰值）
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/experiment.yaml}
NUM_GPUS=${NUM_GPUS:-8}
STAGGER=${STAGGER:-20}
mkdir -p logs
for ((r=0; r<NUM_GPUS; r++)); do
  CUDA_VISIBLE_DEVICES=$r python -m src.generate_shots --config "$CONFIG" \
      --rank $r --world_size $NUM_GPUS "$@" > logs/gen_rank$r.log 2>&1 &
  echo "启动 rank $r (GPU $r)，日志: logs/gen_rank$r.log"
  sleep "$STAGGER"
done
wait
echo "生成完成。查看失败: grep -l Traceback logs/gen_rank*.log"
