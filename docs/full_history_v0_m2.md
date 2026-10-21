# Full-History v0：M2 无缓存模型

M2 已完成可训练的模型核心：每次调用从完整观测前缀重算 H，再计算本次 F，最后运行动作流。所有 Predictor KV 都属于当前计算图，没有模型内部的持久缓存。原版模型和训练 CLI 仍保留原来的默认入口。

## 主要改动

| 文件 | 作用 |
| --- | --- |
| [memory_predictor.py](../src/vjepa_policy/models/memory_predictor.py) | MemoryWorldModelPredictor：完整历史编码、逐历史块状态条件、独立未来查询；复用原 Predictor 参数名 |
| [memory_action_expert.py](../src/vjepa_policy/models/memory_action_expert.py) | MemoryActionExpert：读取完整 H/F KV，提供训练与确定初始噪声的动作采样接口 |
| [memory_policy.py](../src/vjepa_policy/models/memory_policy.py) | FullHistoryVJEPAPolicy：观测/监督编码、输入检查、状态打包、联合损失和无缓存推理；独立 Python 构建入口 |
| [memory_layouts.py](../src/vjepa_policy/models/memory_layouts.py) | 从控制帧索引生成 H/F/A 布局，统一 time-view-height-width 顺序 |
| [action_expert.py](../src/vjepa_policy/models/action_expert.py) | 新增可选有效位与动作 RoPE 参数；原版采样接口额外支持 initial_noise/generator，默认行为保持 |
| [backbone/modules.py](../src/vjepa_policy/models/backbone/modules.py) | 补齐当前 M1 提交缺失的 mask、坐标、prefix KV 入口校验，使原有非法输入测试恢复通过 |

### 历史 H

输入为 `[B,K,P,D_encoder]`，v0 中 P=588。先复用 predictor_embed 和相机 embedding，再按块因果 mask 计算 self-attention。每层 K/V 取自该层 self-attention 输入的投影，与原 MoT 的层对应关系一致。

Cross-attention 临时采用 `[B*K,P,D_predictor]`：每块只读取相同的 whole-task 语言和自己的历史状态 q_i。语言先投影一次，按块复用；状态从 14D 归一化值打包为 96D。状态通过本层 cross-attention 进入 hidden，因此从下一层的 KV 开始生效，不会回流污染较早块。

padding 块输入、隐藏值、返回 KV 均清零，不能作为有效 key。支持原 Predictor 的 activation checkpointing，不 detach 可训练 KV，也不跨 optimizer update 复用。

### 未来 F

`query_future` 只创建当前局部 lattice 中的四组未来查询，v0 共 2352 tokens。每层读取对应层的 H KV 和当前 F KV，使用当前 q_t；最终经 predictor_norm/predictor_proj 得到 JEPA prediction。

F 由 mask token 和相机 embedding 构成，不接收真实未来特征。H 不读取 F；改变查询、当前条件或重复查询均不会修改已有 H KV。局部 `num_patches` 不随历史长度变化。

### 动作 A

动作 Q/K 投影到 Predictor head 空间后使用共享 RoPE；历史和未来 K 已旋转，直接读取。每层 keys 为 H、F、A，动作 chunk 内仍双向联合去噪。历史可见性由 decision_step 决定，未来目标坐标不会扩大历史截止时间。

记忆路径不使用固定 context_len 截断，padding 同时屏蔽 key、query 和 loss。保留原 Flow Matching 方程、action/time embedding 和默认十步采样。一次决策只计算一次 H/F，动作采样循环只更新 A。

`initial_noise` 与 `generator` 互斥。显式噪声需匹配动作形状及上下文 dtype/device，内部复制后使用，便于比较重复调用。底层专家支持 `action_reads_future=False` 的消融接口；v0 policy 会检查该值与固定契约一致，默认仍为 True，正式消融需要后续显式版本/配置支持。

## Python 接口与输入

显式构建入口为 `build_full_history_policy`，生产 v0 会检查 224×224 输入、16px patch、五个 tubelet 的局部 lattice、三视图、96D packed state、32×14 动作、4096D 语言和 V-JEPA2.1 ViT-L adapter。训练脚本的架构选择与 checkpoint schema 分派留到 M3。

下面假定 encoder 已由现有 registry 加载预训练权重；示例随机初始化 Predictor 和 Action Expert，属于 From Scratch 路线：

```python
from vjepa_policy.models.memory_predictor import MemoryWorldModelPredictor
from vjepa_policy.models.memory_policy import build_full_history_policy

predictor = MemoryWorldModelPredictor(
    img_size=224, patch_size=16, num_frames=10, tubelet_size=2,
    embed_dim=encoder.embed_dim, predictor_embed_dim=1024,
    depth=24, num_heads=16, num_views=3,
    proprio_dim=96, lang_dim=4096,
    interpolate_rope=True, canonical_spatial_grid=(16, 16),
    corrected_rope_frequency_pairing=False,
)
policy = build_full_history_policy(encoder, predictor)
loss, metrics = policy(batch)
loss.backward()

policy.eval()
actions = policy.predict_action(observation_batch, initial_noise=noise)
```

旧频率配对的兼容性和时间平移性质沿用 [M1 的说明](full_history_v0_m1.md)。构建器不下载权重，也不把现有双向 Predictor 当作已经训练好的记忆模型。直接实例化模型核心允许缩小空间网格和特征宽度做单元测试；实际 v0 使用上述构建器，并固定语言输入为 `[B,128,4096]`。

batch 必须是字典，且包含 `batch_schema="rmbench_history_v0"`。每行一个 episode、一个决策点；不接受原版八元组。

| 字段 | v0 形状/要求 |
| --- | --- |
| history_state / current_state | [B,K,14] / [B,14]，已按训练统计归一化；current_state 必须等于最后有效历史块状态 |
| history_end_step / history_valid | long [B,K] / bool [B,K]；有效块从 0 开始每 4 步一个，右侧 padding，完整覆盖到 decision_step |
| decision_step | long [B]，非负 16 的倍数 |
| language / language_mask | [B,128,4096] / bool [B,128]，固定 whole-task 文本 |
| history_video 或 history_latents | [B,K,3,3,2,224,224] 或 [B,K,588,D_encoder]，二选一 |
| target_video 或 future_targets | [B,3,3,10,224,224] 或 [B,2352,D_encoder]，训练时二选一 |
| future_valid | bool [B,2352]，训练必需 |
| action / action_is_pad | [B,32,14] / bool [B,32]，训练必需 |

视频已完成 M0 固定的 RGB 预处理。有效观测块按小批独立经过两帧视频 tubelet 编码；padding 视频不送入 Encoder。Target 使用完整十帧编码，额外执行 target LayerNorm 后选择未来四组 tubelet。观测特征不额外执行 target LayerNorm；输入的 future_targets 必须已经完成该归一化，缓存来源检查属于 M3。

Encoder 始终冻结且处于 eval；来自批数据的视觉特征和预计算语言特征不参与梯度更新。观测、目标和语言 padding 即使填有 NaN，也会在对应投影或损失前清除；有效输入中的非有限值被拒绝。

`forward` 返回原 Trainer 形式的 `(loss, metrics)`，metrics 包含 loss、loss_world、loss_action、cos_sim，并保留 forward_dtypes 记录。World loss 为 L1，action loss 为 velocity MSE；先对每个样本的有效 token/动作步分别归一化，再在 batch 上平均。全无效的一项对该样本贡献零。

`predict_action` 只读取观测字段，忽略目标视频、目标特征、未来监督 mask、动作标签和动作 padding；输出为归一化动作。反归一化与控制器执行由后续服务负责。`predict_from_history` 接受调用方提供的 HistoryEncoding、packed current state、语言和 decision_step，计算一次 F 并采样动作；本阶段没有 session 或缓存生命周期管理。

## 验证记录

```bash
python -m unittest discover -s tests -p 'test_memory_v0_model.py' -v
python -m unittest discover -s tests -v
```

CPU policy 环境使用 PyTorch 2.7.1，本次全量 84 项测试全部通过，其中新增 23 项 M2 测试。标准库环境共 37 项，32 项通过，5 个数值测试模块按依赖缺失跳过。日志和汇总保存在被 Git 忽略的 `runs/memory_v0_m2/`。

| 检查 | 已验证的结果 |
| --- | --- |
| 原版 Action Expert | 同 seed 初始化、strict load、训练输出/梯度和采样输出均与冻结的 1ef54c5 实现逐元素一致 |
| Predictor 权重兼容 | 原版与 Memory Predictor 参数名/形状一致，双向 strict load 通过；计算语义仍需要重新训练 |
| H 因果与状态隔离 | 后续观测/状态扰动不改变较早 KV；q_i 从下一层影响对应历史块 |
| H/F 两阶段计算 | 与同样可见性规则下的联合 masked 前向一致；F 的查询不会写回 H |
| 冻结 Encoder | 用真实 V-JEPA2.1 代码的小模型，独立两帧与十帧 context-mask 编码在容差内一致；扰动未来帧不改变 context |
| 视频/特征路径 | 原始预处理视频与其冻结特征输入得到一致 loss；Encoder 和输入特征均无梯度 |
| 联合训练 | 完整小模型前缀的 forward/backward、两次 AdamW update 可运行；world/action loss 均可回传 Predictor |
| 精度与重算 | CPU BF16 autocast 前后向有限；activation checkpointing 的输出与参数梯度一致 |
| 隔离与 padding | 另一 batch episode、额外 padding、无效位置 NaN、监督标签/target 扰动均不污染有效推理 |
| 动作采样 | 固定噪声可重复，形状为 [B,32,14]；H/F 每次决策各运行一次 |
| Loss 权重 | 按样本分别归一化，padding 数量不改变样本权重 |

Action Expert 原有 AdaRMSNorm gate 为零初始化，因此第一次纯 action loss 对 Predictor 的梯度为零；第一次 optimizer update 打开 gate 后，下一次 action loss 可回传到 Predictor。测试覆盖了这个真实更新过程，没有通过修改生产初始化规避它。

模型测试使用三层 Predictor、较小特征宽度和 32px 图像网格，历史覆盖 0、4、8、12、16 的完整前缀，另含短 episode padding；编码等价性测试使用实际 V-JEPA2.1 Transformer 的十二层小宽度实例。本阶段没有加载完整 ViT-L 预训练权重做 GPU 训练或执行模拟器评测，不能据此判断任务成功率。

## 后续边界

M3 接入真实 history dataset/collator、训练入口架构选择、checkpoint schema/恢复和冻结特征缓存，再做真实小批量 update→save→reload。M4 实现持久 KV 缓存和有状态服务。

当前继续使用 M1 的有容量保护的密集参考 mask；长历史超限会明确报错，不会静默截断。完整 full50 最长 81 块的显存/吞吐验证和高效后端属于后续长度实验，本次“小模型完整前缀可运行”不表示生产长序列已达标。
