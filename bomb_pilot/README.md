# 炸弹实验（Bomb Pilot）

> 目的：在投入完整的"导演 × 观众"方法之前，先用 1–2 天验证**最关键的前提**：
> **本地 VLM 扮演的"观众"，能不能感知到"信息给出的顺序"带来的悬念和惊喜差异？**

---

## 1. 实验设计（一图看懂）

每个故事 6 个镜头，做两个版本，**只有第 2 个镜头不同**：

| 位置 | surprise 版（不给观众看危险） | suspense 版（提前给观众看危险） |
|---|---|---|
| 1 | 咖啡馆全景，两人聊天 | **同一个视频文件** |
| 2 | 桌下：一个手提包 | 桌下：**一颗带倒计时的炸弹** |
| 3 | 女人笑着搅咖啡 | **同一个视频文件** |
| 4 | 男人讲笑话 | **同一个视频文件** |
| 5 | 墙上的钟 | **同一个视频文件** |
| 6 | 爆炸 | **同一个视频文件** |

同一个 seed 下，两个版本的公共镜头是**同一个文件**，因此唯一的变量就是第 2 个镜头，这是严格的单变量对照。
一共 3 个故事（咖啡馆炸弹 / 毒酒 / 薄冰），每个故事 3 个 seed，用来证明结果不是挑出来的。

**观众 Agent 做的事**：逐镜头看片，每看完一个镜头回答"你觉得结局会怎样？"（3 个选项：平安无事 / 有危险但脱险 / 有人受伤），然后从模型输出的概率里读出它的"信念"。

**期望看到的结果：**

| 假设 | 期望 |
|---|---|
| **H1 悬念** | 看到炸弹之后到结局之前，suspense 版的悬念曲线**高于** surprise 版 |
| **H2 惊喜** | 看到爆炸那一刻，surprise 版的惊喜**大于** suspense 版（因为 suspense 版的观众早有预期） |
| **H3 传达** | 视频模型真的把炸弹画清楚了（否则 H1 失败不是观众的问题） |
| **H4 对照** | 直接问 VLM"你有多紧张（1-5）"，区分两个版本的效果**不如**基于信念的方法（如果成立，就是论文的一个核心论据） |

---

## 2. 目录结构

```
bomb_pilot/
├── configs/
│   ├── experiment.yaml          # 所有参数：生成模型、观众模型、seed……
│   └── stories/*.json           # 3 个故事（可以照格式加新的）
├── src/
│   ├── generate_shots.py        # 第 1 步：逐镜头生成视频（多卡并行）
│   ├── assemble.py              # 第 2 步：按版本拼成短片
│   ├── run_viewer.py            # 第 3 步：观众 Agent 看片，记录信念曲线
│   ├── analyze.py               # 第 4 步：画图 + 假设检验 + 报告
│   ├── generators.py            # Wan 生成器（diffusers 本地部署）
│   ├── viewer.py                # 观众 Agent（Qwen2.5-VL 本地部署）
│   ├── metrics.py               # 悬念 / 惊喜 / 统计检验
│   ├── common.py, video_io.py   # 工具函数
└── scripts/
    ├── dry_run.sh               # 不用 GPU，先检查流程
    ├── run_generate.sh          # 8 卡并行生成
    ├── run_viewer.sh            # 多卡并行观众评估
    └── run_all.sh               # 一键全流程
```

---

## 3. 环境准备

```bash
conda create -n bomb python=3.10 -y && conda activate bomb
# 按服务器 CUDA 版本装 PyTorch，例如 CUDA 12.1：
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**先确认显存**：运行 `nvidia-smi`。A100 一般是 40G 或 80G，"60G"比较少见，请先确认实际大小。
代码会按显存自动决定要不要 CPU offload（`cpu_offload: auto`），一般不用手动改。

**下载模型**（国内服务器可先 `export HF_ENDPOINT=https://hf-mirror.com`）：

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-14B-Diffusers     # 默认的生成模型
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct         # 默认的观众模型
# 可选：
# huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers  # 画质更好
# huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers   # 速度快，适合调试
# huggingface-cli download Qwen/Qwen2.5-VL-72B-Instruct      # 更强的观众
```

模型文件都比较大（几十 GB 级别），请预留足够的磁盘空间。

---

## 4. 运行

### 第 0 步：Dry run（强烈建议先做，1 分钟）
```bash
bash scripts/dry_run.sh
```
这一步用假模型跑完整个流程，不需要 GPU，也不需要下载模型。报告在 `outputs/dry_run/report/`，里面的数字是随机的，只用来确认环境和代码都正常。

### 第 1 步：生成镜头（8 卡并行）
```bash
bash scripts/run_generate.sh
tail -f logs/gen_rank0.log      # 看进度
```
- 共 3 故事 × 7 个不同镜头 × 3 seed = **63 个片段**，平均每张卡约 8 个
- 中断后直接重跑即可，已生成的片段会自动跳过
- 建议先只开一张卡试一个片段，确认画面没问题：
  `CUDA_VISIBLE_DEVICES=0 python -m src.generate_shots --rank 0 --world_size 63`

### 第 2 步：拼接
```bash
python -m src.assemble
```
完整短片在 `outputs/bomb_pilot/videos/`，**务必自己看几个**，特别是炸弹镜头有没有画出来。

### 第 3 步：观众看片
```bash
bash scripts/run_viewer.sh                       # 7B 观众，8 个进程
# 换 72B：先把配置里 viewer.model_id 改成 Qwen/Qwen2.5-VL-72B-Instruct，然后
GPUS_PER_PROC=4 bash scripts/run_viewer.sh
```

### 第 4 步：分析
```bash
python -m src.analyze
```
输出在 `outputs/bomb_pilot/report/`：
- `report.md`：假设检验结论 + 每个故事的结果 + 传达检查 + "怎么读"的说明
- `<故事>.png`：6 张曲线图（P(出事)、不确定性、悬念、惊喜、直接打分、video vs text 的传达差距）
- `summary.json` / `pairs.csv`：原始数字
- `human_eval_template.csv`：下一步做人工评测时用的打分表

**时间预估**（粗略，取决于模型和步数）：生成每个 5 秒片段约 5–15 分钟，8 卡跑完 63 个片段约 1–2 小时；7B 观众评估约 0.5 小时以内。

---

## 5. 怎么判断结果

| 结果 | 含义 | 下一步 |
|---|---|---|
| H1 ✅ + H2 ✅ | 观众 Agent 能感知信息结构，**idea 前提成立** | 开始做导演 Agent + 闭环 |
| H1 ✅，H4 弱或 ❌ | 最理想：基于信念的测量 > 直接问 VLM | 这张对比图就是论文的动机图 |
| H3 ❌（炸弹没画清楚） | 生成问题，不是观众问题 | 换 `wan22_a14b`，或者改 reveal 镜头的 prompt，让炸弹更醒目 |
| H3 ✅ 但 H1/H2 ❌ | 观众模型不够敏感 | 换 72B 观众、增加 `frames_per_shot`、调整问题措辞；仍不行再考虑换方向 |

**另外两个要看的指标：**
- 日志里的"选项概率质量均值"（`label_mass`）应该 > 0.5。太低说明模型没按"只回答字母"的格式作答，结果不可信
- 3 个故事的方向是否一致：只有一个故事成立的话，说服力不够

---

## 6. 关键实现细节（写论文时会用到）

- **信念怎么读**：不让模型生成文字，而是直接读下一个 token 在 A/B/C 上的概率。选项顺序轮换 3 次后取平均，抵消模型偏爱某个字母的倾向
- **惊喜** = 相邻两个镜头之间信念的变化量（L2 距离）
- **悬念（rollout）** = 让观众先想象 8 种"下一个镜头"，算信念在这些想象下平均会变化多大（参考 Ely, Frankel & Kamenica 2015）
- **不确定性（entropy）** = 信念的归一化熵，是悬念的一个便宜近似
- **text 模式**：观众只读分镜文字、不看视频，代表"计划层"。它和 video 模式的差距 = 画面没传达到的信息

---

## 7. 常见问题

- **显存不足（OOM）**：在配置里把 `min_gpu_mem_gb_no_offload` 调大，强制开启 offload；或者换 `wan22_ti2v_5b`
- **CPU 内存不足**（8 个进程同时加载模型）：调大启动间隔，例如 `STAGGER=120 bash scripts/run_generate.sh`；或者用 `NUM_GPUS=4` 减少进程数
- **`WanPipeline` 报参数错误**：升级 diffusers（`pip install -U diffusers`）。如果预计算文本向量失败，代码会自动回退到普通模式，日志里会有提示
- **Wan2.2 A14B 很慢**：60G 卡上需要 offload，这是正常的；调试阶段建议用 `wan21_14b` 或 `wan22_ti2v_5b`

---

## 8. 加新故事

复制 `configs/stories/cafe_bomb.json`，按同样格式修改，再加到 `experiment.yaml` 的 `stories` 列表里。注意：
- `versions` 里各版本的镜头数必须相同，而且只在 `reveal_index` 这个位置不同
- `prompt` 写给视频模型看（可以用 `{角色名}` 引用角色描述，保证外观一致）；`description` 写给观众读（text 模式用）
- `probe_facts` 里写一个关键信息的是/否问题，用来检查画面有没有传达到
