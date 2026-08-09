# OpenArm2 本机数据与 checkpoint 简表

整理时间：2026-08-05。用途：交接同事远程 debug。

这里只列有效数据集和确实产出 checkpoint 的训练；同一次训练的 checkpoint
合并记录。用户实机观察与文件/日志信息分开描述。

## 主要数据集

数据集根目录：

```text
/home/user/.cache/huggingface/lerobot
```

| 简称 | 完整路径 | Episodes / frames | 说明 |
|---|---|---:|---|
| old-33 | `/home/user/.cache/huggingface/lerobot/thewisp/dddd1` | 33 / 7,702 | 旧基线数据。旧 ACT 基本能抓球并放进盒子；旧 HVLA “fine-ish”，但覆盖不足。 |
| 0803-241 | `/home/user/.cache/huggingface/lerobot/GPU/0803_20260803_174402` | 241 / 40,101 | 用户说明：内容“什么都有”。左臂基本不动。用于原始 0803 HVLA 和 normalization-floor 重训。 |
| 0803-backup-244 | `/home/user/.cache/huggingface/lerobot/GPU/BACKUP_walnut_244ep_used_for_HVLA40k_before_reedit` | 244 / 52,506 | 40k HVLA 实际使用的旧版本。不要误当成现在的 241-episode 目录。 |
| fix-81 | `/home/user/.cache/huggingface/lerobot/fix/0805_20260804_235910` | 81 / 15,403 | 用户说明：核桃和盒子的相对位置固定。左臂基本不动。 |
| LandR-66 | `/home/user/.cache/huggingface/lerobot/LandR/0805_20260805_144658` | 66 / 9,352 | 用户说明：录制时特意动了左手。左臂关节覆盖明显增大。 |

这些主要数据集都是双臂、16 维 position action、48 维
position/velocity/torque state、三路相机、30 FPS。

补充：部分数据集删除过 episode，但历史 health/split sidecar 没同步收缩。
远程脚本应以当前 `meta/info.json`、episode metadata 和 Parquet 为准，不要仅按
旧 sidecar 的行数判断 episode 数量。

## 已产出 checkpoint 的训练

训练根目录：

```text
/home/user/.cache/lerobot/runs
```

| Run | 模型 / 数据 | 本次训练产出的 checkpoints | 关键信息与实机观察 |
|---|---|---|---|
| `e793d02ed576` | ACT / old-33 | 5k、10k、15k、20k | 旧基线；20k 路径为 `output/checkpoints/020000/pretrained_model`。用户反馈基本能完成抓球放盒。chunk size 100，执行 100 步。 |
| `4bea5f9b78ad` | HVLA / old-33 | 30k、40k、50k | 从旧 run 的 20k 继续训练；用户反馈 “fine-ish”，主要问题是数据覆盖不足。建议比较用 50k。 |
| `ea6f1a0dc25c` | HVLA / 0803-backup-244 | 每 4k 一个，直到 40k | 日志实际读入 52,506 frames，因此来源是 backup-244，不是当前 241 数据。 |
| `1ac4f1f291c7` | HVLA / 0803-241 | 5k、10k、15k、20k、25k、30k | 原始问题模型。用户反馈 30k 会卡住或只在同一位置附近移动。训练时没有 normalization floor，旧版 target 还存在跨 episode 边界污染。 |
| `c6d1d6430eb1` | HVLA / fix-81 | 每 500 步一个，直到 10k | 用户/同事反馈某些 checkpoint 推理动作很猛烈，曾需要 E-stop。没有 normalization floor，也带旧版边界污染。 |
| `6ed528d71ff2` | HVLA / 0803-241 | 5k、10k、15k、20k、25k、30k | normalization-floor 重训，state position std floor 为 `0.5°`。数值异常得到抑制，但 30k 实机仍不理想；interval=5 时一度更像是在找球。旧版边界污染仍在训练权重中。 |
| `95dad5554c9a` | ACT / fix-81 | 每 1k 一个，直到 25k | 新 ACT。chunk size 100，执行 20 步。用户反馈实机主要表现为突然横移，明显差于旧 ACT。 |
| `1b2cf91ac941` | HVLA / LandR-66 | 每 1k 一个，直到 10k | 已完成；包含 normalization floor 和 episode-boundary 修复，仍是 absolute action。验证集 loss 在约 3k 最低，之后持续变差，所以优先比较 2k、3k、4k 与 10k，不要默认 10k 最好。 |

所有 checkpoint 的完整路径规则为：

```text
/home/user/.cache/lerobot/runs/<RUN_ID>/output/checkpoints/<CHECKPOINT>/pretrained_model
```

HVLA 的目录名通常是 `checkpoint-30000`；ACT 的目录名可能是 `020000`。

## 远程 debug 最值得先看的对照

1. 旧 ACT `e793d02ed576` 20k 对比新 ACT `95dad5554c9a` 25k：两者实机表现差异最大。
2. 原始 HVLA `1ac4f1f291c7` 30k 对比 normalization 重训
   `6ed528d71ff2` 30k：可分离 normalization floor 的影响，但二者都含旧边界问题。
3. 最新 LandR `1b2cf91ac941`：优先离线比较 2k/3k/4k/10k 的 action chunk；它是这批关键训练里同时含 boundary 修复和 validation 的 run。
4. `fix-81` 和 `LandR-66` 不是严格 A/B：除了左臂是否运动，episode 数、任务覆盖和采集过程也不同。

当前最重要的两个已知风险：

- 新 ACT 在两次已记录推理的首帧上，按训练统计归一化后最大 state 偏差约为
  `18,083σ–27,021σ`，集中在几乎不动的左臂关节；旧 ACT 对相近现场状态约为
  `3.4σ`。这很可能解释新 ACT 的突然横移，但仍应通过同帧离线 action chunk
  对比确认。
- 早期 HVLA 训练样本会把 episode 末尾的 future action 错接到下一条 episode。
  0803、fix 的旧 checkpoint 都已经学入该问题；推理时不能补救。LandR run
  `1b2cf91ac941` 已使用修复后的 loader。

## 日志与进一步说明

主要 GUI/推理日志：

```text
/home/user/projects/lerobot-openarm2-consolidate/outputs/hvla_runs
/home/user/projects/lerobot-openarm2-consolidate/outputs/record
```

更完整的因果分析、action chunk 证据和代码来源见：

```text
/home/user/projects/lerobot-openarm2-consolidate/docs/investigations/hvla_flow_s1_2026-08-05.md
```

当前机器人配置：

```text
/home/user/.config/lerobot/robots/openarm2_dual_verified.json
```

本次整理只写了这份 Markdown，没有修改训练/推理后端，也没有启动或停止进程。
