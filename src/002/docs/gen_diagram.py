"""生成 Excalidraw 原理图 - 自动计算文字框高度, 杜绝截断; 用箭头连接流程。

用法: python gen_diagram.py
输出: IK原理图.excalidraw
"""
import json, random

W = {"elements": []}
_seed = [1000]
def nid():
    _seed[0] += 1
    return f"e{_seed[0]}"

# 中文按字号全宽, 英文/数字半宽; 估算文字块高度
def text_dims(text, fs, lh=1.3):
    lines = text.split("\n")
    n = len(lines)
    height = int(n * fs * lh) + 8
    # 宽度: 最长行
    maxw = 0
    for ln in lines:
        w = 0
        for ch in ln:
            w += fs * (1.0 if ord(ch) > 0x2000 else 0.6)
        maxw = max(maxw, w)
    return int(maxw) + 10, height

def rect(x, y, w, h, stroke, bg):
    e = {"id": nid(), "type": "rectangle", "x": x, "y": y, "width": w, "height": h,
         "angle": 0, "strokeColor": stroke, "backgroundColor": bg, "fillStyle": "solid",
         "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1, "opacity": 100,
         "groupIds": [], "frameId": None, "roundness": {"type": 3}, "seed": random.randint(1, 99999),
         "version": 1, "versionNonce": 1, "isDeleted": False, "boundElements": [],
         "updated": 1, "link": None, "locked": False}
    W["elements"].append(e)
    return e

def text(x, y, s, fs=14, color="#1e1e1e", font=1, align="left"):
    w, h = text_dims(s, fs)
    e = {"id": nid(), "type": "text", "x": x, "y": y, "width": w, "height": h,
         "angle": 0, "strokeColor": color, "backgroundColor": "transparent", "fillStyle": "solid",
         "strokeWidth": 1, "strokeStyle": "solid", "roughness": 1, "opacity": 100,
         "groupIds": [], "frameId": None, "roundness": None, "seed": random.randint(1, 99999),
         "version": 1, "versionNonce": 1, "isDeleted": False, "boundElements": [],
         "updated": 1, "link": None, "locked": False, "fontSize": fs, "fontFamily": font,
         "text": s, "textAlign": align, "verticalAlign": "top", "containerId": None,
         "originalText": s, "lineHeight": 1.3}
    W["elements"].append(e)
    return e

# 带文字的卡片: 自动按文字高度撑开框, pad留白
def card(x, y, minw, title, body, stroke, bg, fs=13, pad=14):
    tw, th = text_dims(title, fs+2)
    bw, bh = text_dims(body, fs)
    w = max(minw, tw + 2*pad, bw + 2*pad)
    h = th + bh + 2*pad + 6
    rect(x, y, w, h, stroke, bg)
    text(x+pad, y+pad, title, fs+2, stroke)
    text(x+pad, y+pad+th+4, body, fs, "#1e1e1e", font=3)
    return w, h

def arrow(x1, y1, x2, y2, color="#495057", both=False):
    e = {"id": nid(), "type": "arrow", "x": x1, "y": y1, "width": x2-x1, "height": y2-y1,
         "angle": 0, "strokeColor": color, "backgroundColor": "transparent", "fillStyle": "solid",
         "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1, "opacity": 100,
         "groupIds": [], "frameId": None, "roundness": {"type": 2}, "seed": random.randint(1, 99999),
         "version": 1, "versionNonce": 1, "isDeleted": False, "boundElements": [],
         "updated": 1, "link": None, "locked": False, "points": [[0, 0], [x2-x1, y2-y1]],
         "lastCommittedPoint": None, "startBinding": None, "endBinding": None,
         "startArrowhead": "arrow" if both else None, "endArrowhead": "arrow"}
    W["elements"].append(e)
    return e

def build():
    # ===== 标题 =====
    text(420, -70, "SO-101 逆运动学原理图 (IK)", 34, "#1971c2")
    text(420, -20, "解析 / 数值 / 神经网络 / 混合法 —— 公式推导与算法逻辑", 16, "#868e96")

    # ===== ① FK 与 IK 互为反问题 (箭头连接) =====
    text(40, 40, "① 核心问题：FK 与 IK 互为反问题", 20, "#1971c2")
    card(40, 78, 380, "正运动学 FK  (容易·唯一解)",
         "关节角 θ\n───────►\n末端位姿 (x,y,z,a,psi)", "#2f9e44", "#ebfbee")
    card(720, 78, 380, "逆运动学 IK  (难·多解) 本项目",
         "末端位姿\n───────►\n关节角 theta", "#e8590c", "#fff4e6")
    arrow(430, 130, 715, 130, "#2f9e44")
    arrow(715, 175, 430, 175, "#e8590c")
    text(490, 106, "FK 正向", 13, "#2f9e44")
    text(480, 180, "IK 反向(求解)", 13, "#e8590c")

    # ===== ② FK 公式推导 (三步, 箭头串联) =====
    text(40, 250, "② 正运动学 FK 公式推导 (三步连乘)", 20, "#2f9e44")
    y2 = 290
    w1, h1 = card(40, y2, 300, "第1步·单关节变换",
                 "Ti = Trans(origin)\n   · Rot_fixed(rpy)\n   · Rot(axis, thetai)", "#2f9e44", "#ebfbee")
    w2, h2 = card(400, y2, 330, "第2步·绕轴旋转(罗德里格斯)",
                 "R(a,t) = I\n  + sin(t)·[a]x\n  + (1-cos(t))·[a]x^2", "#2f9e44", "#ebfbee")
    w3, h3 = card(790, y2, 320, "第3步·连乘得末端",
                 "T = T1·T2·T3·T4·T5·Ttool\n位置 p = T[0:3,3]\n姿态 R = T[0:3,0:3]", "#2f9e44", "#ebfbee")
    arrow(340, y2+40, 398, y2+40)
    arrow(730, y2+40, 788, y2+40)
    yc = y2 + max(h1, h2, h3) + 20
    _, hpit = card(40, yc, 1070, "⚠ FK 踩过的两个坑 (已修复)",
         "坑A 关节顺序: URDF字典序反、含gripper、漏pan -> 必须按 base到末端 链 [pan,lift,elbow,wrist_flex,wrist_roll]\n"
         "坑B 末端把手: gripper_frame 有 98mm, FK漏算导致把手砸桌 -> 已在末尾补 Ttool 偏移", "#e03131", "#fff5f5")

    _build_methods(yc + hpit)  # 传入坑卡片的实际底部y

def _build_methods(prev_bottom):
    # ===== ③ 四种 IK 方法 (2x2 卡片) =====
    # prev_bottom 是上一区块卡片的底部y, 标题留60余量
    y3 = prev_bottom + 60
    text(40, y3-34, "③ 四种 IK 方法的公式推导", 20, "#1971c2")
    _, ha = card(40, y3, 530, "方法1·解析法 (几何闭式解)",
        "余弦定理求肘角 θ₃:\n  cosθ₃ = (r²+z²-L₁²-L₂²)/(2·L₁·L₂)\n  θ₃ = ±arccos(…)  ← ±给肘上/肘下(多解)\n肩: θ₂ = atan2(z,r) - atan2(L₂sinθ₃, L₁+L₂cosθ₃)\n底座: θ₁ = atan2(y,x)\n\n✓极快  ✗需精确连杆参数\n本项目: 参数不准 → 4/50成功, 243mm", "#e8590c", "#fff4e6")
    _, hb = card(600, y3, 520, "方法2·数值法 (阻尼最小二乘DLS)",
        "误差 e = x_target − x_current\n雅可比 J: ẋ = J·θ̇\n迭代(避奇异): Δθ = Jᵀ(JJᵀ+λ²I)⁻¹·e\n  θ ← θ + Δθ, 直到 ‖e‖<ε\nλ自适应: 近奇异点加大阻尼\n\n✓通用·最准(0.9mm)  ✗慢(~15ms)依赖初值", "#7048e8", "#f3f0ff")
    yr2 = y3 + max(ha, hb) + 20
    _, hc = card(40, yr2, 530, "方法3·神经网络 (端到端MLP) ★重点",
        "学习映射 f: (p,α,ψ, q_cur) → q_target\n输入12维: [x,y,z, sinα,cosα, sinψ,cosψ, q₁..₅]\n  (角度用sin/cos解决周期性)\n网络: 12→256→256→128→5\n限位硬约束(tanh缩放,永不超限):\n  q = (q_hi+q_lo)/2 + (q_hi−q_lo)/2·tanh(z)\n\n✓极快(0.5ms)不需参数  ✗精度受数据量限", "#0c8599", "#e3fafc")
    _, hd = card(600, yr2, 520, "方法4·混合法 (NN初值+数值微调) ★最优",
        "θ_final = DLS_refine( f_NN(x), N=3步 )\n\n① NN 给一个很接近的初值(~10mm)\n② 数值法从好初值出发, 只需2-3步微调\n③ 收敛到亚毫米\n\n为什么最优: 好初值→迭代极少\n→ 兼具NN速度 + 数值法精度\n结果: 软件中位0.9mm, 比纯数值快2倍", "#2f9e44", "#ebfbee")

    # ===== ④ 数据流水线 (4框箭头串联) =====
    y4 = yr2 + max(hc, hd) + 75
    text(40, y4-34, "④ 神经网络训练数据流水线 (方案A: 真机数据驱动)", 20, "#7048e8")
    pw = 250; gap = 35; ph = 110
    xs = 40
    p1w,_ = card(xs, y4, pw, "1·真机采集(600)",
        "随机安全姿态\n→机械臂移动\n→读实际关节角 q\n→FK算末端位姿", "#1971c2", "#e7f5ff")
    xs2 = xs + pw + gap
    p2w,_ = card(xs2, y4, pw, "2·数据转换(×5)",
        "q_target = q_real\n(p,α,ψ)=FK_task(q)\nq_cur=随机无关姿态★\nx=[p,sinα,cosα,sinψ,cosψ,q_cur]", "#7048e8", "#f3f0ff")
    xs3 = xs2 + pw + gap
    p3w,_ = card(xs3, y4, pw, "3·训练MLP",
        "Loss = MSE(q_pred,\n  q_target) + 限位惩罚\nAdam + 早停\n→ ik_mlp_best.pth", "#0c8599", "#e3fafc")
    xs4 = xs3 + pw + gap
    card(xs4, y4, pw, "4·验证",
        "软件往返 10.9mm\n真机到达 34mm\n重复性 0.70°", "#2f9e44", "#ebfbee")
    ay = y4 + ph/2
    arrow(xs+p1w, ay, xs2, ay)
    arrow(xs2+p2w, ay, xs3, ay)
    arrow(xs3+p3w, ay, xs4, ay)
    # 捷径陷阱红条
    card(xs2, y4+ph+40, 760, "★ 关键教训: q_cur 的构造决定成败",
        "若 q_cur=[目标+小噪声] → 网络学会\"抄初值\"捷径, 没真学映射 → 真机误差112mm!\n"
        "改用 q_cur=[随机无关姿态] → 逼网络从位姿真学映射 → 真机34mm", "#e03131", "#fff5f5")

    _build_error(y4+ph+40)

def _build_error(prev_y):
    y5 = prev_y + 190
    text(40, y5-34, "⑤ 真机误差分解 & 闭环补偿 (关键结论)", 20, "#e8590c")
    _, he = card(40, y5, 530, "核心发现: 真机误差 ≠ 软件误差",
        "软件层: 数值0.9 / 神经15 / 混合0.9 mm (差16倍)\n真机层: 三方法全是 35~36 mm (几乎一样!)\n\n为什么? 主体是机械误差(~33mm):\n  · 肩部重力下垂 ~4.6°\n  · 开环控制 + 齿轮间隙\n→ IK再准也被机械误差淹没\n→ 结论: 仿真精度 ≠ 真机精度", "#e8590c", "#fff4e6")
    _, hf = card(600, y5, 520, "如何降误差? 关节空间闭环 (不需相机!)",
        "✗ FK闭环失败: 实际位置也是FK算的,观测不到偏差\n✓ 关节闭环成功: 编码器给真实关节角反馈\n  迭代: servo_cmd += 0.8·(q_target − q_actual)\n  命令→读实际→补差→再命令, 3~5次收敛\n\n结果: 33.6mm → 4.0mm (改善88%)\n  点5轨迹: 51→5.8→3.8→1.7mm", "#2f9e44", "#ebfbee")

def save():
    out = {"type": "excalidraw", "version": 2, "source": "https://excalidraw.com",
           "elements": W["elements"],
           "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"}, "files": {}}
    with open("IK原理图.excalidraw", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"✓ 已生成 IK原理图.excalidraw, 元素数={len(W['elements'])}")

if __name__ == "__main__":
    build()
    save()
