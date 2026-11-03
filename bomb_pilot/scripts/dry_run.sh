#!/usr/bin/env bash
# 不需要 GPU / 不下载模型：用假生成器和假观众把整个流程跑一遍，检查环境和代码是否正常。
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.generate_shots --dry_run
python -m src.assemble --dry_run
python -m src.run_viewer --dry_run
python -m src.analyze --dry_run
echo "Dry run 完成，报告见 outputs/dry_run/report/report.md（数字是随机的，只用来检查流程）"
