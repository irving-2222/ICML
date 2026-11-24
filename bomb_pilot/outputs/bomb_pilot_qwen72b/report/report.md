# 炸弹实验报告

悬念指标：`rollout`；每个配对 = 同一故事、同一 seed 下的两个版本。

## 假设检验（汇总所有故事）

| 假设 | 含义 | 平均差值 | 效应量 dz | 单侧 p | 正向比例 | 结论 |
|---|---|---|---|---|---|---|
| H1 | 窗口内悬念：suspense − surprise | 0.113 | 0.60 | 0.059 | 0.67 | 🟡 方向对但不显著 |
| H2 | 结局惊喜：surprise − suspense | 0.545 | 1.65 | 0.002 | 1.00 | ✅ 支持 |
| H4（对照） | 直接打分：suspense − surprise | 1.687 | 2.04 | 0.002 | - | ✅ 支持 |

## 分故事结果

| 故事 | seeds | H1 平均差 | H2 平均差 | H4 平均差 | 曲线图 |
|---|---|---|---|---|---|
| cafe_bomb | 3 | 0.284 | 0.941 | 2.631 | [cafe_bomb.png](cafe_bomb.png) |
| poisoned_wine | 3 | -0.118 | 0.195 | 1.099 | [poisoned_wine.png](poisoned_wine.png) |
| thin_ice | 3 | 0.171 | 0.499 | 1.331 | [thin_ice.png](thin_ice.png) |

## H3 传达检查（关键信息是否真的被画出来）

p_yes_expected 应该高（>0.5），p_yes_other 应该低。若 reveal 镜头的 p_yes_expected 低，说明视频模型没把炸弹画清楚——这时 H1 不成立不是观众模型的问题，而是生成问题。

| 故事 | 版本 | seed | 事实 | p_yes(应出现的镜头) | p_yes(其他镜头) |
|---|---|---|---|---|---|
| cafe_bomb | surprise | 0 | bomb_visible | - | 0.002 |
| cafe_bomb | surprise | 0 | explosion_visible | 0.998 | 0.000 |
| cafe_bomb | surprise | 1 | bomb_visible | - | 0.000 |
| cafe_bomb | surprise | 1 | explosion_visible | 0.998 | 0.000 |
| cafe_bomb | surprise | 2 | bomb_visible | - | 0.001 |
| cafe_bomb | surprise | 2 | explosion_visible | 0.999 | 0.000 |
| cafe_bomb | suspense | 0 | bomb_visible | 0.993 | 0.002 |
| cafe_bomb | suspense | 0 | explosion_visible | 0.998 | 0.000 |
| cafe_bomb | suspense | 1 | bomb_visible | 0.998 | 0.001 |
| cafe_bomb | suspense | 1 | explosion_visible | 0.998 | 0.000 |
| cafe_bomb | suspense | 2 | bomb_visible | 0.988 | 0.002 |
| cafe_bomb | suspense | 2 | explosion_visible | 0.999 | 0.000 |
| poisoned_wine | surprise | 0 | poison_visible | - | 0.000 |
| poisoned_wine | surprise | 0 | collapse_visible | 0.914 | 0.002 |
| poisoned_wine | surprise | 1 | poison_visible | - | 0.000 |
| poisoned_wine | surprise | 1 | collapse_visible | 0.920 | 0.001 |
| poisoned_wine | surprise | 2 | poison_visible | - | 0.000 |
| poisoned_wine | surprise | 2 | collapse_visible | 0.991 | 0.003 |
| poisoned_wine | suspense | 0 | poison_visible | 0.977 | 0.000 |
| poisoned_wine | suspense | 0 | collapse_visible | 0.914 | 0.002 |
| poisoned_wine | suspense | 1 | poison_visible | 0.691 | 0.000 |
| poisoned_wine | suspense | 1 | collapse_visible | 0.920 | 0.001 |
| poisoned_wine | suspense | 2 | poison_visible | 0.634 | 0.000 |
| poisoned_wine | suspense | 2 | collapse_visible | 0.991 | 0.003 |
| thin_ice | surprise | 0 | crack_visible | - | 0.002 |
| thin_ice | surprise | 0 | fall_visible | 0.998 | 0.001 |
| thin_ice | surprise | 1 | crack_visible | - | 0.095 |
| thin_ice | surprise | 1 | fall_visible | 0.515 | 0.016 |
| thin_ice | surprise | 2 | crack_visible | - | 0.100 |
| thin_ice | surprise | 2 | fall_visible | 0.965 | 0.003 |
| thin_ice | suspense | 0 | crack_visible | 0.996 | 0.001 |
| thin_ice | suspense | 0 | fall_visible | 0.998 | 0.001 |
| thin_ice | suspense | 1 | crack_visible | 0.998 | 0.102 |
| thin_ice | suspense | 1 | fall_visible | 0.515 | 0.016 |
| thin_ice | suspense | 2 | crack_visible | 0.995 | 0.105 |
| thin_ice | suspense | 2 | fall_visible | 0.965 | 0.003 |

## 传达差距（video vs text 信念的 L2 距离）

差距大 = 观众看视频得到的信息和读分镜文字得到的不一样（画面没传达到计划的信息）。

| 故事 | 版本 | seed | 平均差距 | reveal 镜头处差距 |
|---|---|---|---|---|
| cafe_bomb | surprise | 0 | 0.090 | 0.010 |
| cafe_bomb | surprise | 1 | 0.091 | 0.008 |
| cafe_bomb | surprise | 2 | 0.084 | 0.019 |
| cafe_bomb | suspense | 0 | 0.193 | 0.100 |
| cafe_bomb | suspense | 1 | 0.186 | 0.090 |
| cafe_bomb | suspense | 2 | 0.289 | 0.266 |
| poisoned_wine | surprise | 0 | 0.544 | 0.812 |
| poisoned_wine | surprise | 1 | 0.515 | 0.756 |
| poisoned_wine | surprise | 2 | 0.536 | 0.805 |
| poisoned_wine | suspense | 0 | 0.176 | 0.095 |
| poisoned_wine | suspense | 1 | 0.347 | 0.382 |
| poisoned_wine | suspense | 2 | 0.288 | 0.187 |
| thin_ice | surprise | 0 | 0.157 | 0.125 |
| thin_ice | surprise | 1 | 0.178 | 0.081 |
| thin_ice | surprise | 2 | 0.200 | 0.126 |
| thin_ice | suspense | 0 | 0.396 | 0.191 |
| thin_ice | suspense | 1 | 0.538 | 0.356 |
| thin_ice | suspense | 2 | 0.512 | 0.284 |

## 怎么读这份报告

- **H1 + H2 都支持** → 观众 Agent 能感知信息结构带来的悬念/惊喜差异，idea 的关键前提成立。
- **H1 支持、H4 不支持或更弱** → 最理想：说明"基于信念"的测量比"直接问 VLM"更好，这是论文的一个核心论据。
- **H3 失败** → 先换更强的生成模型（如 wan22_a14b）或改 reveal 镜头 prompt，再看 H1。
- **全不支持且 H3 通过** → 观众模型本身不行：试 72B 观众、增加抽帧、调整问题措辞；仍不行则考虑换方向。
- 若日志里 `label_mass` 很低（<0.5），说明模型没按"只回答字母"的格式作答，需要调整 prompt。