# Full-History v0：M0 输入契约

M0 为全历史 JEPA-WAM 固定输入与版本契约，首个验证任务为 RMBench put_back_block。模型版本是 `v0`，序列化 schema 版本是 `1`；这与 Python 包原有的 `0.1.0` 版本独立。M0 尚未实现历史 Predictor、增量 KV、v0 训练或 v0 服务。

## 兼容性与基线

原版仍使用原来的模型类、八元组 batch、训练和服务入口。M0 不修改现有运行文件，也不向原入口添加一个尚不能工作的 v0 开关。缺少记忆版本字段的旧 checkpoint 继续属于 legacy。

[legacy_baseline.json](../configs/memory_v0/legacy_baseline.json) 固定基线提交 `031a7c7d74a71b3bd97b12e85d8a6ae84d252eec` 和 103 个原版 Python/Shell 运行文件的 SHA-256。v0 在 `full-history-v0` 分支开发，数据、环境和权重位置通过被忽略的 local.env 指定，不复制大型资源。

新增 [memory_config.py](../src/vjepa_policy/memory_config.py) 提供不可变契约、规范化 JSON、指纹、前缀索引和时间坐标。`contract_from_checkpoint_metadata` 是给后续新 loader 使用的解析器，当前不接管原版 loader。完整 v0 元数据必须显式包含 `policy_arch=full_history`、`model_version=v0` 和完整 `memory_schema`；不完整或未知的记忆标记报错，不能静默当作 legacy。

## 固定的输入

机器可读的单一配方见 [input_contract.json](../configs/memory_v0/input_contract.json)。M0 固定一套配方；改变相机、采样、位置或归一化语义时，必须显式修订契约和验证，而不是继续沿用原指纹。

| 项目 | v0 M0 约定 |
| --- | --- |
| 任务与示范 | put_back_block / demo_clean / aloha_agilex / 原生 v1.0 HDF5 |
| 相机 | cam_head、cam_left_wrist、cam_right_wrist，顺序固定、独立编码 |
| RGB | 官方 decode_image_bit；uint8 240×320×3；不旋转、不额外换通道 |
| 模型图像 | 共用 VideoClipTransform，stretch 到 224×224，bilinear+antialias；mean/std 都是 (0.5,0.5,0.5) |
| Encoder | 冻结 vjepa2_1_vitl，patch=16、tubelet=2 |
| 历史块 | 截止 i，取 [max(0,i−4),i]；i=0,4,8,…；第 0 块为 [0,0] |
| 决策 | t=0,16,32,…；从 episode 起点提供全部历史，不裁掉旧前缀 |
| 未来目标 | t+[-4,0,4,8,12,16,20,24,28,32]；仅负索引重复首帧 |
| 尾部 | 必须 t+32 < episode_frames；不完整未来窗口不生成训练样本 |
| 动作标签 | 原始 action[t:t+32]，已经对齐下一保存状态，额外 shift=0 |
| 状态/动作 | 14D：[左臂6、左夹爪、右臂6、右夹爪]；绝对关节弧度 |
| 夹爪 | 索引 6/13，0=closed、1=open；仅夹爪输出裁剪，不把关节角裁剪到 [-1,1] |
| 归一化 | 原训练集合统计，state/action 都为 QUANTILE，clip_values=false |
| 状态打包 | 14D 归一化状态 → max_state_dim=48 → 数值+有效位共 96D |
| 文本 | 固定 whole-task 指令，经原 DEFAULT_PROMPT；T5 特征 128×4096 |
| 禁止条件 | stage/subtask、原始目标垫身份、未来图像、监督 target latent |
| Token 数 | 历史块 588；单决策未来查询 2352 |
| 状态条件 | 每块绑定自身 q_i；当前 future/action 使用 q_t |
| 时间 | 数据帧/控制调用序数，固定单位 8；不随当前历史长度重新编号 |
| RoPE 时间规则 | H_i=i/8；未来 tubelet=(t+8/16/24/32)/8；动作 j=(t+j+1)/8 |

原生动作已经对应下一状态；动作的位置坐标使用 t+j+1，并不意味着再移动动作标签。初始重复帧只解决缺少过去观测，不允许用重复未来帧替代完整未来监督。

native_qpos 是正式控制协议，dense15 是数据节拍诊断，必须分开记录。一次 TOPP 控制调用不等于固定物理时长；本版本使用序数时间，不将其称为秒。controller 属于实验记录，不因切换 controller 而修改已经保存的输入契约。

## 首个任务的选择

put_back_block 要求先将方块移至中心，按按钮，再放回最初的垫子。最初位置是历史线索，所需资产、50 条示范、原版控制/回放检查已存在，适合作为 M0 单任务验证。后续增加 2–3 任务对照属于后续实验扩展，不能据此宣称全部 RMBench 任务已完成适配。

本次检查了头部相机前 8 条示范的 t=0 图像，使用与模型相同的 resize/normalize 变换；四种初始垫位置均可辨认。该结果只证明这些样例的线索被采样和保留，不证明冻结 JEPA 特征已经编码该线索，也不证明策略已学会记忆。

## 使用方式

从仓库根目录运行。仅检查契约无需导入训练依赖：

```bash
python scripts/check_memory_v0.py
python -m unittest discover -s tests -p 'test_memory_v0_contract.py' -v
```

核验真实数据需要现有 policy 环境，使用新的报告目录：

```bash
source configs/memory_v0.local.env
"$POLICY_PYTHON" scripts/check_memory_v0.py \
  --manifest "$RMBENCH_PREPARED_DATA/manifest.json" \
  --controller native_qpos \
  --output "$POLICY_SOURCE_ROOT/runs/memory_v0_m0/new_input_audit"
```

检查器校验 codec/示范文件哈希、状态和动作的所有时间步、三相机长度、版本/节拍、归一化统计来源、固定指令和全部 v0 采样 RGB。默认解码每 4 帧的实际采样位置；`--decode-all` 额外检查非采样帧。`results.json` 会写明覆盖范围。错误时保留失败报告和已有进度，不覆盖旧目录。

检查器不运行模拟器或训练。生成的 `initial_cue_preview.png` 需要视觉复核，检查器不会自动宣称“线索可见”或“模型可训练”。

本地共享环境可能仍以 editable 方式指向旧机械盘源码。local.env 将 `PYTHONPATH` 的首项设为当前仓库的 `src` 和根目录；检查器也明确使用当前源码。不要重新安装或改动正在被其他实验使用的共享环境。

## 本次验收记录

| 检查 | 结果 |
| --- | --- |
| 标准库环境测试 | 35 项，32 项通过，3 个数值测试模块因缺少依赖跳过 |
| 完整 policy 环境测试 | 41 项全部通过，包含新增 8 项契约测试 |
| full50 数据 | 50 episodes、17,562 frames、1,013 个合法决策样本 |
| 历史长度 | 最长训练前缀 81 块，即 47,628 tokens |
| RGB | 所有 v0 采样位置、三路 RGB 解码通过 |
| 动作/状态 | 所有示范 next-state 对齐最大误差 0，无二次 shift |
| 归一化来源 | full50 使用对应 50 条训练示范的统计 |
| 初始线索 | 前 8 条示范的变换后头部图像已视觉复核；原始位置可辨 |
| 原版运行源码 | 相对记录的 103 个运行文件，M0 字节未改动 |

这些结果保存在本机 `runs/memory_v0_m0/`；该目录被 Git 忽略。`m0_summary.json` 汇总检查、源文件指纹和复用的旧结果来源。复用的原版模拟器结果包括 native expert 通过，以及 4 条示范在 native_qpos/dense15 下分别回放成功；原版推理检查记录了 train/serve 图像一致和有限 32×14 输出。这些是已存在的证据，本次没有重跑 GPU 推理或模拟器，不能当作 v0 模型验证。

81 块只是本数据采样得到的训练长度，不是缓存容量上限。若在线允许 1,000 次控制调用，按 4 步写入将最多需要 251 块；后续需单独验证该长度、显存和训练分布。M0 不设置静默历史截断。

## 下一阶段边界

M1 再实现 block-causal mask、共享 RoPE 和 prefix attention；M2 实现可训练的历史/未来/动作路径；M3 接入新版 dataset、构建器与 checkpoint loader。保持原版默认入口，明确选择 v0 后才进入新架构。当前不能使用尚未实现的 `--policy-arch full_history` 启动训练。
