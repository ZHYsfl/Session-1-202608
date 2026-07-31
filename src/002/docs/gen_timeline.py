"""生成调试历程时间线图 - 箭头串联的节点, 自动框高杜绝截断。
用法: python gen_timeline.py  →  调试历程图.excalidraw
"""
import json, random

W = {"elements": []}
_s = [2000]
def nid():
    _s[0] += 1
    return f"t{_s[0]}"

def tdims(text, fs, lh=1.3):
    lines = text.split("\n")
    h = int(len(lines)*fs*lh)+8
    mw = 0
    for ln in lines:
        w = sum(fs*(1.0 if ord(c) > 0x2000 else 0.6) for c in ln)
        mw = max(mw, w)
    return int(mw)+10, h

def rect(x,y,w,h,st,bg,sw=2):
    W["elements"].append({"id":nid(),"type":"rectangle","x":x,"y":y,"width":w,"height":h,"angle":0,
        "strokeColor":st,"backgroundColor":bg,"fillStyle":"solid","strokeWidth":sw,"strokeStyle":"solid",
        "roughness":1,"opacity":100,"groupIds":[],"frameId":None,"roundness":{"type":3},
        "seed":random.randint(1,99999),"version":1,"versionNonce":1,"isDeleted":False,
        "boundElements":[],"updated":1,"link":None,"locked":False})

def ellipse(x,y,w,h,st,bg):
    W["elements"].append({"id":nid(),"type":"ellipse","x":x,"y":y,"width":w,"height":h,"angle":0,
        "strokeColor":st,"backgroundColor":bg,"fillStyle":"solid","strokeWidth":2,"strokeStyle":"solid",
        "roughness":1,"opacity":100,"groupIds":[],"frameId":None,"roundness":None,
        "seed":random.randint(1,99999),"version":1,"versionNonce":1,"isDeleted":False,
        "boundElements":[],"updated":1,"link":None,"locked":False})

def text(x,y,s,fs=14,color="#1e1e1e",font=1,align="left"):
    w,h = tdims(s,fs)
    W["elements"].append({"id":nid(),"type":"text","x":x,"y":y,"width":w,"height":h,"angle":0,
        "strokeColor":color,"backgroundColor":"transparent","fillStyle":"solid","strokeWidth":1,
        "strokeStyle":"solid","roughness":1,"opacity":100,"groupIds":[],"frameId":None,"roundness":None,
        "seed":random.randint(1,99999),"version":1,"versionNonce":1,"isDeleted":False,"boundElements":[],
        "updated":1,"link":None,"locked":False,"fontSize":fs,"fontFamily":font,"text":s,
        "textAlign":align,"verticalAlign":"top","containerId":None,"originalText":s,"lineHeight":1.3})
    return h

def arrow(x1,y1,x2,y2,color="#868e96",sw=3):
    W["elements"].append({"id":nid(),"type":"arrow","x":x1,"y":y1,"width":x2-x1,"height":y2-y1,"angle":0,
        "strokeColor":color,"backgroundColor":"transparent","fillStyle":"solid","strokeWidth":sw,
        "strokeStyle":"solid","roughness":1,"opacity":100,"groupIds":[],"frameId":None,"roundness":{"type":2},
        "seed":random.randint(1,99999),"version":1,"versionNonce":1,"isDeleted":False,"boundElements":[],
        "updated":1,"link":None,"locked":False,"points":[[0,0],[x2-x1,y2-y1]],"lastCommittedPoint":None,
        "startBinding":None,"endBinding":None,"startArrowhead":None,"endArrowhead":"arrow"})

# 一个历程节点: 序号圆 + 标题 + 正文, 框自动撑高
NODE_X = 150
NODE_W = 940
def node(num, y, title, body, stroke, bg, sw=2, numbg=None):
    tw, th = tdims(title, 16)
    bw, bh = tdims(body, 14)
    pad = 14
    h = th + bh + 2*pad + 6
    # 序号圆
    ellipse(90, y+18, 34, 34, stroke, numbg or bg)
    text(100, y+23, str(num), 16, "#ffffff" if numbg else stroke)
    # 框
    rect(NODE_X, y, NODE_W, h, stroke, bg, sw)
    text(NODE_X+pad, y+pad, title, 16, stroke)
    text(NODE_X+pad, y+pad+th+4, body, 14, "#1e1e1e", font=1)
    return h

def build():
    text(340, -70, "SO-101 调试突破历程 (一步步怎么解决的)", 32, "#c2255c")
    text(340, -22, "每个节点: 🔴症状 → 🔍诊断 → 🟢解决 → ✅效果   |   竖向箭头 = 时间推进", 15, "#868e96")

    nodes = [
        (1,"坑1·FK关节顺序错误","🔴 FK算出末端 Z=-53mm (末端在基座下方,物理荒谬)\n🔍 URDF字典序是反的、含gripper、漏了shoulder_pan → 读URDF父子链确认正确链序\n🟢 按 base→末端 重建链序 [pan,lift,elbow,wrist_flex,wrist_roll]\n✅ Z 从 -53mm → +268mm (合理)","#e03131","#fff5f5"),
        (2,"坑2·数值IK 求解失败(无解)","🔍 根因就是坑1 —— FK错→目标位姿在错误坐标系→不可达\n🟢 FK修好后,目标位姿正确,数值IK自然收敛\n✅ 数值IK 往返误差 0.8mm, 成功率 49/50","#f08c00","#fff9db"),
        (3,"坑3·数据采集时把手一直砸桌 (用户喊停)","🔴 我以为末端Z=157mm安全, 实际把手梆梆砸桌面\n🔍 FK只算到关节,漏了末端把手98mm长度! 旧过滤下真实尖端最低到 Z=-83mm\n🟢 解析URDF gripper_frame_joint(98mm), FK末尾补 T_tool 偏移\n✅ FK现在算的是真实把手尖端,不再是关节位置","#e03131","#fff5f5"),
        (4,"坑4·补了把手偏移还是砸桌","🔍 我一直用软件模型\"猜\"桌面高度,不知道桌面在机械臂坐标系的真实位置(在赌博)\n🟢 手动标定: 关扭力把把手轻触桌面,读关节角算出桌面真实 Z=66mm (标准差仅1mm!)\n     安全线 = 桌面66 + 重力下垂 + 裕度 = Z≥192mm, 移动前先经HOME中转\n✅ 静态验证5个最低点, 把手离桌~10cm, 用户确认零碰撞","#f08c00","#fff9db"),
        (5,"坑5·神经网络真机误差高达112mm ★最重要的算法突破","🔴 验证损失0.003很漂亮,软件误差23mm,但真机到达差112mm; 重复性9°(很散)\n🔍 训练数据 q_current=目标+小噪声 → 网络发现\"抄初值\"捷径,没真学位姿映射!\n     验证损失低是假象,真机上初值(HOME)远离目标就露馅\n🟢 q_current改成\"随机无关姿态\",逼网络从位姿学映射; 数据×5增强\n✅ 软件23→11mm, 重复性 9°→0.70°(高度一致!), 真机 112→34mm","#e03131","#fff5f5",3),
        (6,"发现6·真机34mm误差到底来自哪? (对照实验)","🔍 同批目标点测三方法: 软件层 数值0.9/神经15/混合0.9mm(差16倍), 真机层全是35~36mm!\n     再测: 直接命令关节角、完全不经IK, 真机仍偏33.5mm(肩部下垂4.6°)\n💡 结论: 真机误差主体是机械误差(~33mm), 与IK方法无关 → \"仿真精度≠真机精度\"","#1971c2","#e7f5ff"),
        (7,"突破7·把机械误差也降下来 —— 关节空间闭环 🏆 终极突破","🔴 先试FK闭环→失败(只降<2mm): 因为\"实际位置\"也是FK算的,观测不到真实物理偏差\n💡 想通: 编码器能读真实关节角! 改用关节空间闭环\n🟢 迭代: servo_cmd += 0.8·(q_target − q_actual)  命令→读实际→补差→再命令, 3~5次收敛\n✅ 真机 33.6mm → 4.0mm (改善88%!)  点5轨迹: 51→5.8→3.8→1.7mm  且不需相机","#2f9e44","#d3f9d8",3),
    ]
    y = 40
    prev_cy = None
    for tup in nodes:
        num,title,body,st,bg = tup[0],tup[1],tup[2],tup[3],tup[4]
        sw = tup[5] if len(tup)>5 else 2
        numbg = st if num in (5,7) else None
        # 上一节点到本节点的箭头
        if prev_cy is not None:
            arrow(107, prev_cy, 107, y-4)
        h = node(num, y, title, body, st, bg, sw, numbg)
        prev_cy = y + h
        y += h + 45

    # 底部总结: 误差下降 + 方法论
    y += 10
    text(150, y-30, "总结", 20, "#495057")
    _summary_prog(150, y)
    _summary_lesson(620, y)
    save()

def _summary_prog(x,y):
    body="神经网络首训:  112 mm   (抄初值捷径)\n        ↓ 修数据构造\nrandom训练:     34 mm   (真学映射)\n        ↓ 关节空间闭环\n闭环补偿后:      4 mm   (编码器反馈)\n\n总共降低 96% ✅"
    tw,th=tdims("📉 真机误差一路下降",16); bw,bh=tdims(body,14); pad=14
    rect(x,y,440,th+bh+2*pad+6,"#2f9e44","#ebfbee")
    text(x+pad,y+pad,"📉 真机误差一路下降",16,"#2f9e44")
    text(x+pad,y+pad+th+4,body,14,"#1e1e1e")

def _summary_lesson(x,y):
    body="1. 物理荒谬值是最好的报警器\n   (Z=-53mm 一眼看出FK错了)\n\n2. 软件指标漂亮 ≠ 真机可用\n   (损失0.003却真机112mm; 务必真机验证)\n\n3. 别靠模型猜, 要标定/测量真实量\n   (桌面高度、机械误差都要实测)"
    tw,th=tdims("🎓 三条方法论(贯穿全程)",16); bw,bh=tdims(body,14); pad=14
    rect(x,y,470,th+bh+2*pad+6,"#7048e8","#f3f0ff")
    text(x+pad,y+pad,"🎓 三条方法论(贯穿全程)",16,"#7048e8")
    text(x+pad,y+pad+th+4,body,14,"#1e1e1e")

def save():
    out={"type":"excalidraw","version":2,"source":"https://excalidraw.com","elements":W["elements"],
         "appState":{"gridSize":None,"viewBackgroundColor":"#ffffff"},"files":{}}
    json.dump(out, open("调试历程图.excalidraw","w",encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ 已生成 调试历程图.excalidraw, 元素数={len(W['elements'])}")

if __name__=="__main__":
    build()
