"""Render the uncompressed Full-History architecture on CPU with Pillow.

Layout reference: https://arxiv.org/html/2609.37250, Figure 1.
Model semantics: memory_predictor.py, memory_action_expert.py, memory_policy.py.
Run this file directly; it writes a high-resolution PNG beside the script.
"""

from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT, SCALE = 2680, 1620, 2
OUT = Path(__file__).with_name("full_history_v0_architecture.png")
im = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), "white")
d = ImageDraw.Draw(im)

C = {
    "ink": "#263345", "muted": "#64748B", "line": "#7C899B",
    "blue": "#547DB7", "blue_bg": "#ECF2FA", "blue_light": "#B0C5E2",
    "purple": "#8154B4", "purple_bg": "#F4EFFA", "purple_light": "#BEA3DA",
    "orange": "#C07B3F", "orange_bg": "#FCF2E8", "orange_light": "#E7B88F",
    "green": "#558777", "green_bg": "#EAF4EF", "pink": "#A26C88",
    "pink_bg": "#F7EDF3", "grey": "#F4F6F9", "mask": "#E9EDF2",
    "rose": "#A45970", "rose_bg": "#FBF0F3",
}


def font(size, bold=False):
    weight = "Bold" if bold else "Regular"
    candidates = [
        Path(f"/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc"),
        Path(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), round(size * SCALE))
    raise RuntimeError("Install Noto Sans CJK to render the Chinese annotations.")


def text(x, y, s, size=25, color="ink", bold=False, anchor="mm"):
    d.text((round(x * SCALE), round(y * SCALE)), s, font=font(size, bold),
           fill=C.get(color, color), anchor=anchor)


def rect(box, fill="white", edge=None, radius=14, width=2):
    coords = tuple(round(v * SCALE) for v in box)
    d.rounded_rectangle(coords, radius=round(radius * SCALE), fill=C.get(fill, fill),
                        outline=C.get(edge, edge) if edge else None,
                        width=round(width * SCALE))


def line(points, color="line", width=2, dashed=False):
    pts = [(x * SCALE, y * SCALE) for x, y in points]
    if not dashed:
        d.line(pts, fill=C.get(color, color), width=round(width * SCALE), joint="curve")
        return
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        distance = math.hypot(x2 - x1, y2 - y1)
        for start in range(0, round(distance), 18 * SCALE):
            end = min(start + 10 * SCALE, distance)
            if distance:
                d.line([(x1 + (x2 - x1) * start / distance,
                         y1 + (y2 - y1) * start / distance),
                        (x1 + (x2 - x1) * end / distance,
                         y1 + (y2 - y1) * end / distance)],
                       fill=C.get(color, color), width=round(width * SCALE))


def arrow(points, color="line", width=2.5, dashed=False, head=10):
    line(points, color, width, dashed)
    x, y = points[-1]
    px, py = points[-2]
    angle = math.atan2(y - py, x - px)
    points = [(x, y),
              (x - head * math.cos(angle) + head * .5 * math.sin(angle),
               y - head * math.sin(angle) - head * .5 * math.cos(angle)),
              (x - head * math.cos(angle) - head * .5 * math.sin(angle),
               y - head * math.sin(angle) + head * .5 * math.cos(angle))]
    d.polygon([(round(a * SCALE), round(b * SCALE)) for a, b in points], fill=C.get(color, color))


def dashed_box(box, color="line"):
    x0, y0, x1, y1 = box
    line([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], color, dashed=True)


def tokens(x, y, n, color, hollow=False, size=27, gap=10):
    for i in range(n):
        left = x + i * (size + gap)
        rect((left, y, left + size, y + size), "white" if hollow else color,
             edge=color, radius=3, width=1.8)


def block(box, title, subtitle=None, color="purple", size=27):
    rect(box, f"{color}_bg", color, radius=8, width=1.7)
    x0, y0, x1, y1 = box
    cy = (y0 + y1) / 2
    text((x0 + x1) / 2, cy - (15 if subtitle else 0), title, size, color, True)
    if subtitle:
        text((x0 + x1) / 2, cy + 22, subtitle, 21, "muted")


# Title and compact legend.
text(60, 60, "Full-History V-JEPA Policy", 43, bold=True, anchor="lm")
rect((818, 35, 1140, 88), "blue_bg", radius=27)
text(979, 60, "v0 · 未压缩架构", 25, "blue", True)
text(60, 118, "完整历史视觉记忆  +  当前未来查询  +  Flow-Matching 动作生成", 27, "muted", anchor="lm")
line([(60, 152), (2620, 152)], "mask")

# Frozen target encoder, used exclusively for training supervision.
dashed_box((60, 185, 1230, 347))
text(84, 213, "仅训练：未来潜变量监督", 25, "blue", True, "lm")
rect((83, 254, 290, 321), "grey", "line", radius=8)
text(186, 287, "完整目标视频片段", 22)
arrow([(290, 288), (334, 288)], "blue")
block((335, 250, 689, 324), "V-JEPA 2.1 Encoder", "同一冻结编码器 · ViT-L", "blue", 25)
arrow([(690, 288), (731, 288)], "blue")
tokens(744, 270, 6, "blue_light", size=24, gap=8)
text(836, 314, "LN + 选择未来位置", 18, "muted")
arrow([(950, 288), (991, 288)], "blue")
block((996, 252, 1207, 323), "L1 loss", "L_world", "purple", 25)

# Text/state conditioning: explicit conditioning semantics, no ambiguous bus.
rect((1270, 185, 2050, 347), "grey", radius=13)
text(1293, 213, "语言与机器人状态条件", 25, bold=True, anchor="lm")
block((1293, 246, 1645, 323), "Instruction → T5 Encoder", "冻结 T5 · language tokens", "pink", 23)
block((1671, 246, 2027, 323), "Robot state → Projection", "可训练 · 每块 q_i / 当前 q_t", "green", 23)

rect((2100, 185, 2620, 347), "grey", radius=13)
text(2128, 215, "图例", 25, bold=True, anchor="lm")
rect((2128, 249, 2153, 274), "blue_bg", "blue", radius=4)
text(2170, 262, "冻结编码器", 22, "blue", anchor="lm")
rect((2390, 249, 2415, 274), "purple_bg", "purple", radius=4)
text(2431, 262, "可训练模块", 22, "purple", anchor="lm")
line([(2128, 308), (2171, 308)], "line", dashed=True)
text(2186, 308, "仅训练监督", 22, "muted", anchor="lm")

# Main repeated transformer structure, bottom to top as in paper Figure 1.
rect((60, 468, 1360, 1004), "purple_bg", "purple", radius=18, width=2.5)
text(83, 500, "Future Predictor", 30, "purple", True, "lm")
text(1335, 500, "24 层 · H / F 两阶段共享权重", 24, "purple", anchor="rm")
line([(710, 549), (710, 969)], "#DDCFEA", width=1.7, dashed=True)
text(386, 552, "历史 H", 27, "blue", True)
text(1030, 552, "当前未来查询 F", 27, "purple", True)

rect((1450, 468, 2050, 1004), "orange_bg", "orange", radius=18, width=2.5)
text(1473, 500, "Action Expert", 30, "orange", True, "lm")
text(2026, 500, "24 层", 24, "orange", anchor="rm")
text(1750, 552, "动作 A · Flow Matching", 27, "orange", True)

# Read from bottom to top: QKV -> attention -> cross attention -> FFN.
for left, right, center, color, condition in [
        (83, 687, 385, "blue", "language + 历史块状态 q_i"),
        (733, 1337, 1035, "purple", "language + 当前状态 q_t"),
        (1473, 2027, 1750, "orange", "language + 当前状态 q_t")]:
    block((left, 590, right, 644), "FFN", color=color)
    block((left, 665, right, 745), "Cross Attention", condition, color, 25)
    block((left, 922, right, 976), "QKV Projection", color=color, size=25)
    arrow([(center, 665), (center, 647)], color, width=2)
    arrow([(center, 922), (center, 895)], color, width=2)
    arrow([(center, 775), (center, 748)], color, width=2)

# Layerwise attention interfaces are within the repeated blocks, not final outputs.
rect((83, 775, 2027, 895), "white", "line", radius=8, width=2)
rect((92, 784, 684, 886), "blue_bg", radius=5)
rect((735, 784, 1337, 886), "purple_bg", radius=5)
rect((1473, 784, 2018, 886), "orange_bg", radius=5)
text(385, 814, "H_i 读取 H_0 … H_i", 29, "blue", True)
text(385, 858, "块间因果 · 块内双向", 22, "blue")
text(1035, 814, "F 读取 [H KV, F KV]", 28, "purple", True)
text(1035, 858, "H → F 单向读取", 22, "purple")
text(1750, 814, "A 读取 [H KV, F KV, A KV]", 25, "orange", True)
text(1750, 858, "动作 chunk 内联合去噪", 22, "orange")
arrow([(685, 828), (732, 828)], "blue", width=3, head=10)
arrow([(1338, 828), (1470, 828)], "purple", width=3, head=11)
rect((1359, 789, 1449, 811), "white", radius=0)
text(1404, 800, "H + F", 18, "purple", True)
text(1404, 863, "逐层 KV", 17, "muted")

# Outputs and losses: F predicts latents, A predicts velocity.
arrow([(1035, 468), (1035, 452)], "purple", width=2.6)
rect((832, 394, 1235, 449), "white", radius=0)
tokens(858, 406, 9, "purple_light", size=24, gap=10)
text(813, 419, "未来潜变量预测", 24, "purple", anchor="rm")
arrow([(1100, 395), (1100, 325)], "purple", width=2.1, dashed=True)

arrow([(1750, 468), (1750, 450)], "orange", width=2.6)
rect((1517, 393, 1973, 449), "white", radius=0)
tokens(1537, 405, 10, "orange_light", size=24, gap=10)
text(1750, 369, "速度预测 vθ  /  积分得到动作", 24, "orange")

# Full-resolution token inputs.
arrow([(385, 1065), (385, 979)], "blue", width=2.6)
rect((83, 1023, 687, 1078), "white", radius=0)
tokens(105, 1037, 5, "blue_light", size=25, gap=9)
text(306, 1049, "…", 30, "blue")
tokens(343, 1037, 5, "blue_light", size=25, gap=9)
rect((543, 1031, 679, 1068), "blue_bg", "blue", radius=6)
text(611, 1049, "当前 H_t", 21, "blue", True)
text(385, 1102, "H_0, H_1, …, H_t：episode 起点到当前", 24, "blue")

arrow([(1035, 1065), (1035, 979)], "purple", width=2.6)
rect((734, 1023, 1336, 1078), "white", radius=0)
tokens(792, 1037, 12, "purple_light", hollow=True, size=25, gap=13)
text(1035, 1102, "Learnable future queries", 25, "purple")
text(1035, 1143, "4 × 3 × 14 × 14 = 2352 tokens", 23, "purple")
text(1035, 1182, "未来位置：t+8 / +16 / +24 / +32", 22, "muted")
text(1035, 1222, "每次决策重新生成 F", 23, "muted")

arrow([(1750, 1065), (1750, 979)], "orange", width=2.6)
rect((1473, 1023, 2027, 1078), "white", radius=0)
tokens(1550, 1037, 11, "orange_light", hollow=True, size=25, gap=9)
text(1750, 1102, "Noisy action tokens  aτ  +  flow time τ", 23, "orange")
text(1750, 1143, "训练：aτ = (1−τ) ε + τ a_demo", 22, "muted")
text(1750, 1182, "推理：高斯噪声 → 10 步积分 → 32×14 动作", 21, "muted")
text(1750, 1222, "执行前 16 步，然后重新决策", 23, "orange")

# Encoder and three-view observation input, kept small to emphasize architecture.
rect((60, 1140, 684, 1285), "grey", radius=12)
for xx, label in [(83, "Head"), (168, "L wrist"), (253, "R wrist")]:
    rect((xx, 1162, xx + 69, 1211), "white", "line", radius=6, width=1.5)
    rect((xx + 22, 1174, xx + 47, 1199), "blue_bg", "blue", radius=11, width=1.5)
    text(xx + 34, 1231, label, 16, "muted")
arrow([(333, 1188), (371, 1188)], "blue")
block((376, 1157, 665, 1238), "V-JEPA 2.1 Encoder", "冻结 · 每视角独立编码", "blue", 22)
text(372, 1263, "每 4 步一个双帧块 → 3 × 14 × 14 = 588 tokens / 块", 19, "blue")
arrow([(520, 1156), (520, 1123), (700, 1123), (700, 1049), (682, 1049)], "blue", width=2)

# Attention visibility matrix, showing block groups rather than individual patches.
rect((2100, 468, 2620, 1285), "grey", radius=16)
text(2360, 508, "Attention visibility", 27, bold=True)
text(2360, 547, "行 = Query，列 = Key / Value", 21, "muted")
labels = ["H_0", "H_1", "H_t", "F", "A"]
mx, my, step, cell = 2237, 626, 65, 53
for col, label in enumerate(labels):
    text(mx + col * step + cell / 2, my - 34, label, 24, "muted")
for row, label in enumerate(labels):
    text(mx - 41, my + row * step + cell / 2, label, 24, "muted", anchor="rm")
    color = "blue_light" if row < 3 else "purple_light" if row == 3 else "orange_light"
    for col in range(5):
        rect((mx + col * step, my + row * step,
              mx + col * step + cell, my + row * step + cell),
             color if col <= row else "mask", radius=4)
text(2360, 988, "H：块间因果，块内双向", 22, "blue")
text(2360, 1031, "F：读取全部已观测 H 和当前 F", 22, "purple")
text(2360, 1074, "A：读取全部 H、当前 F 和 A", 22, "orange")
line([(2130, 1112), (2590, 1112)], "#DCE3EB")
text(2360, 1155, "当前 context 已包含在最后一个 H 块", 21, "ink", True)
text(2360, 1194, "H 保留完整 14×14 空间网格", 22, "blue")
text(2360, 1233, "同色方格表示允许读取", 21, "muted")

# Training / inference separation and persistent history lifecycle.
rect((60, 1324, 1115, 1535), "rose_bg", "#DCBAC6", radius=15)
text(84, 1359, "联合训练", 27, "rose", True, "lm")
text(84, 1407, "L = L_world (L1) + L_action (velocity MSE)", 27, "rose", True, "lm")
text(84, 1453, "每个样本重算完整历史；Predictor 与 Action Expert 联合反向传播。", 22, anchor="lm")
text(84, 1495, "Action loss 经逐层 H/F KV 回传 Predictor；视觉与文本编码器冻结。", 22, "muted", anchor="lm")

rect((1155, 1324, 2620, 1535), "blue_bg", "#BBCEE4", radius=15)
text(1179, 1359, "在线推理：逐层历史 KV 缓存", 27, "blue", True, "lm")
block((1180, 1396, 1478, 1456), "保留旧 H KV", color="blue", size=23)
text(1509, 1426, "+", 30, "blue", True)
block((1543, 1396, 1864, 1456), "编码新增真实观测", color="blue", size=23)
arrow([(1878, 1426), (1932, 1426)], "blue", width=2.8)
block((1944, 1396, 2290, 1456), "追加为完整 H KV", color="blue", size=23)
arrow([(2304, 1426), (2350, 1426)], "blue", width=2.8)
block((2362, 1396, 2595, 1456), "查询 F → 生成 A", color="purple", size=21)
text(1179, 1495, "重新决策保留 H；F / A 为本次临时量；episode 结束才清空历史。", 23, "blue", anchor="lm")

text(60, 1580, "布局参考：V-JEPA Policy, Fig. 1 · arXiv:2609.37250  |  结构依据：本地 Full-History v0 / M2–M4 实现", 19, "muted", anchor="lm")
text(2620, 1580, "三视角 RMBench 配置 · 未压缩版本", 19, "muted", anchor="rm")

OUT.parent.mkdir(parents=True, exist_ok=True)
im.save(OUT, dpi=(300, 300), optimize=True)
print(f"Saved {OUT} ({im.width} × {im.height})")
