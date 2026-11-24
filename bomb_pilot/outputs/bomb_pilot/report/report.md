# 炸弹实验报告

悬念指标：`rollout`；每个配对 = 同一故事、同一 seed 下的两个版本。

## 假设检验（汇总所有故事）

| 假设 | 含义 | 平均差值 | 效应量 dz | 单侧 p | 正向比例 | 结论 |
|---|---|---|---|---|---|---|
| H1 | 窗口内悬念：suspense − surprise | 0.044 | 1.02 | 0.010 | 0.89 | ✅ 支持 |
| H2 | 结局惊喜：surprise − suspense | 0.424 | 1.00 | 0.002 | 1.00 | ✅ 支持 |
| H4（对照） | 直接打分：suspense − surprise | 0.629 | 1.43 | 0.002 | - | ✅ 支持 |

## 分故事结果

| 故事 | seeds | H1 平均差 | H2 平均差 | H4 平均差 | 曲线图 |
|---|---|---|---|---|---|
| cafe_bomb | 3 | 0.086 | 0.975 | 1.135 | [cafe_bomb.png](cafe_bomb.png) |
| poisoned_wine | 3 | 0.010 | 0.142 | 0.155 | [poisoned_wine.png](poisoned_wine.png) |
| thin_ice | 3 | 0.036 | 0.155 | 0.596 | [thin_ice.png](thin_ice.png) |

## H3 传达检查（关键信息是否真的被画出来）

p_yes_expected 应该高（>0.5），p_yes_other 应该低。若 reveal 镜头的 p_yes_expected 低，说明视频模型没把炸弹画清楚——这时 H1 不成立不是观众模型的问题，而是生成问题。

| 故事 | 版本 | seed | 事实 | p_yes(应出现的镜头) | p_yes(其他镜头) |
|---|---|---|---|---|---|
| cafe_bomb | surprise | 0 | bomb_visible | - | 0.024 |
| cafe_bomb | surprise | 0 | explosion_visible | 0.986 | 0.001 |
| cafe_bomb | surprise | 1 | bomb_visible | - | 0.042 |
| cafe_bomb | surprise | 1 | explosion_visible | 0.969 | 0.007 |
| cafe_bomb | surprise | 2 | bomb_visible | - | 0.057 |
| cafe_bomb | surprise | 2 | explosion_visible | 0.985 | 0.002 |
| cafe_bomb | suspense | 0 | bomb_visible | 0.993 | 0.028 |
| cafe_bomb | suspense | 0 | explosion_visible | 0.986 | 0.005 |
| cafe_bomb | suspense | 1 | bomb_visible | 0.996 | 0.051 |
| cafe_bomb | suspense | 1 | explosion_visible | 0.969 | 0.010 |
| cafe_bomb | suspense | 2 | bomb_visible | 0.992 | 0.068 |
| cafe_bomb | suspense | 2 | explosion_visible | 0.985 | 0.006 |
| poisoned_wine | surprise | 0 | poison_visible | - | 0.011 |
| poisoned_wine | surprise | 0 | collapse_visible | 0.718 | 0.021 |
| poisoned_wine | surprise | 1 | poison_visible | - | 0.015 |
| poisoned_wine | surprise | 1 | collapse_visible | 0.777 | 0.016 |
| poisoned_wine | surprise | 2 | poison_visible | - | 0.031 |
| poisoned_wine | surprise | 2 | collapse_visible | 0.827 | 0.045 |
| poisoned_wine | suspense | 0 | poison_visible | 0.982 | 0.010 |
| poisoned_wine | suspense | 0 | collapse_visible | 0.718 | 0.024 |
| poisoned_wine | suspense | 1 | poison_visible | 0.990 | 0.014 |
| poisoned_wine | suspense | 1 | collapse_visible | 0.777 | 0.016 |
| poisoned_wine | suspense | 2 | poison_visible | 0.990 | 0.012 |
| poisoned_wine | suspense | 2 | collapse_visible | 0.827 | 0.045 |
| thin_ice | surprise | 0 | crack_visible | - | 0.065 |
| thin_ice | surprise | 0 | fall_visible | 0.987 | 0.025 |
| thin_ice | surprise | 1 | crack_visible | - | 0.149 |
| thin_ice | surprise | 1 | fall_visible | 0.955 | 0.015 |
| thin_ice | surprise | 2 | crack_visible | - | 0.281 |
| thin_ice | surprise | 2 | fall_visible | 0.988 | 0.062 |
| thin_ice | suspense | 0 | crack_visible | 0.963 | 0.061 |
| thin_ice | suspense | 0 | fall_visible | 0.987 | 0.033 |
| thin_ice | suspense | 1 | crack_visible | 0.998 | 0.160 |
| thin_ice | suspense | 1 | fall_visible | 0.955 | 0.021 |
| thin_ice | suspense | 2 | crack_visible | 0.975 | 0.257 |
| thin_ice | suspense | 2 | fall_visible | 0.988 | 0.138 |

## 传达差距（video vs text 信念的 L2 距离）

差距大 = 观众看视频得到的信息和读分镜文字得到的不一样（画面没传达到计划的信息）。

| 故事 | 版本 | seed | 平均差距 | reveal 镜头处差距 |
|---|---|---|---|---|
| cafe_bomb | surprise | 0 | 0.057 | 0.101 |
| cafe_bomb | surprise | 1 | 0.072 | 0.106 |
| cafe_bomb | surprise | 2 | 0.060 | 0.110 |
| cafe_bomb | suspense | 0 | 0.104 | 0.032 |
| cafe_bomb | suspense | 1 | 0.091 | 0.072 |
| cafe_bomb | suspense | 2 | 0.083 | 0.064 |
| poisoned_wine | surprise | 0 | 0.579 | 0.569 |
| poisoned_wine | surprise | 1 | 0.606 | 0.544 |
| poisoned_wine | surprise | 2 | 0.783 | 0.814 |
| poisoned_wine | suspense | 0 | 0.389 | 0.169 |
| poisoned_wine | suspense | 1 | 0.384 | 0.259 |
| poisoned_wine | suspense | 2 | 0.427 | 0.358 |
| thin_ice | surprise | 0 | 0.056 | 0.050 |
| thin_ice | surprise | 1 | 0.135 | 0.068 |
| thin_ice | surprise | 2 | 0.141 | 0.082 |
| thin_ice | suspense | 0 | 0.294 | 0.167 |
| thin_ice | suspense | 1 | 0.354 | 0.204 |
| thin_ice | suspense | 2 | 0.427 | 0.252 |

## 怎么读这份报告

- **H1 + H2 都支持** → 观众 Agent 能感知信息结构带来的悬念/惊喜差异，idea 的关键前提成立。
- **H1 支持、H4 不支持或更弱** → 最理想：说明"基于信念"的测量比"直接问 VLM"更好，这是论文的一个核心论据。
- **H3 失败** → 先换更强的生成模型（如 wan22_a14b）或改 reveal 镜头 prompt，再看 H1。
- **全不支持且 H3 通过** → 观众模型本身不行：试 72B 观众、增加抽帧、调整问题措辞；仍不行则考虑换方向。
- 若日志里 `label_mass` 很低（<0.5），说明模型没按"只回答字母"的格式作答，需要调整 prompt。