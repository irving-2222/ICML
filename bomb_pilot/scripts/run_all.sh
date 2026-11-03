#!/usr/bin/env bash
# 一键跑完整流程：生成 -> 拼接 -> 观众 -> 分析
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/experiment.yaml}
bash scripts/run_generate.sh
python -m src.assemble --config "$CONFIG"
bash scripts/run_viewer.sh
python -m src.analyze --config "$CONFIG"
echo "全部完成，报告见 outputs/bomb_pilot/report/report.md"
