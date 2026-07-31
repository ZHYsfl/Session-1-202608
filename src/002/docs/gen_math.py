"""生成数形结合的IK数学推导图 - 带坐标系/连杆/三角形/角度弧/数字算例。
用法: python gen_math.py  →  IK数学推导图.excalidraw
"""
import json, math, random

W = {"elements": []}
_s = [3000]
def nid():
    _s[0] += 1
    return f"m{_s[0]}"

def tdims(text, fs, lh=1.3):
    lines = text.split("\n")
    h = int(len(lines)*fs*lh)+8
    mw = max((sum(fs*(1.0 if ord(c) > 0x2000 else 0.6) for c in ln) for ln in lines), default=0)
    return int(mw)+10, h

def rect(x,y,w,h,st,bg,sw=2,dash=False):
    W["elements"].append({"id":nid(),"type":"rectangle","x":x,"y":y,"width":w,"height":h,"angle":0,
        "strokeColor":st,"backgroundColor":bg,"fillStyle":"solid","strokeWidth":sw,
        "strokeStyle":"dashed" if dash else "solid","roughness":1,"opacity":100,"groupIds":[],
        "frameId":None,"roundness":{"type":3},"seed":random.randint(1,99999),"version":1,
        "versionNonce":1,"isDeleted":False,"boundElements":[],"updated":1,"link":None,"locked":False})

def txt(x,y,s,fs=14,color="#1e1e1e",font=1):
    w,h = tdims(s,fs)
    W["elements"].append({"id":nid(),"type":"text","x":x,"y":y,"width":w,"height":h,"angle":0,
        "strokeColor":color,"backgroundColor":"transparent","fillStyle":"solid","strokeWidth":1,
        "strokeStyle":"solid","roughness":1,"opacity":100,"groupIds":[],"frameId":None,"roundness":None,
        "seed":random.randint(1,99999),"version":1,"versionNonce":1,"isDeleted":False,"boundElements":[],
        "updated":1,"link":None,"locked":False,"fontSize":fs,"fontFamily":font,"text":s,
        "textAlign":"left","verticalAlign":"top","containerId":None,"originalText":s,"lineHeight":1.3})
    return h

def line(pts, color="#1e1e1e", sw=2, dash=False, arrow_end=False):
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    x0,y0=pts[0]
    W["elements"].append({"id":nid(),"type":"line" if not arrow_end else "arrow","x":x0,"y":y0,
        "width":max(xs)-min(xs),"height":max(ys)-min(ys),"angle":0,"strokeColor":color,
        "backgroundColor":"transparent","fillStyle":"solid","strokeWidth":sw,
        "strokeStyle":"dashed" if dash else "solid","roughness":1,"opacity":100,"groupIds":[],
        "frameId":None,"roundness":{"type":2},"seed":random.randint(1,99999),"version":1,
        "versionNonce":1,"isDeleted":False,"boundElements":[],"updated":1,"link":None,"locked":False,
        "points":[[p[0]-x0,p[1]-y0] for p in pts],"lastCommittedPoint":None,
        "startBinding":None,"endBinding":None,"startArrowhead":None,
        "endArrowhead":"arrow" if arrow_end else None})

def dot(x,y,color="#e03131",r=6):
    W["elements"].append({"id":nid(),"type":"ellipse","x":x-r,"y":y-r,"width":2*r,"height":2*r,"angle":0,
        "strokeColor":color,"backgroundColor":color,"fillStyle":"solid","strokeWidth":2,
        "strokeStyle":"solid","roughness":1,"opacity":100,"groupIds":[],"frameId":None,"roundness":None,
        "seed":random.randint(1,99999),"version":1,"versionNonce":1,"isDeleted":False,
        "boundElements":[],"updated":1,"link":None,"locked":False})

def arc(cx,cy,r,a0,a1,color="#7048e8",sw=2):
    # 用多段线近似圆弧, a0/a1为度; excalidraw y向下, 故用-sin
    pts=[]
    n=max(6,int(abs(a1-a0)/10))
    for i in range(n+1):
        a=math.radians(a0+(a1-a0)*i/n)
        pts.append([cx+r*math.cos(a), cy-r*math.sin(a)])
    line(pts,color,sw)

def panel(x,y,w,h,title,stroke,bg):
    rect(x,y,w,h,stroke,bg,2)
    txt(x+14,y+10,title,17,stroke)

def save():
    out={"type":"excalidraw","version":2,"source":"https://excalidraw.com","elements":W["elements"],
         "appState":{"gridSize":None,"viewBackgroundColor":"#ffffff"},"files":{}}
    json.dump(out, open("IK数学推导图.excalidraw","w",encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"OK IK数学推导图.excalidraw elements={len(W['elements'])}")

def build():
    txt(360,-70,"SO-101 逆运动学 数学推导 (数形结合)",30,"#c2255c")
    txt(360,-24,"每个角度是什么·公式怎么来·拿真实数字算一遍",15,"#868e96")

    # ===== 面板1: 俯视图求 θ1 =====
    panel(40,40,540,360,"① 底座旋转 θ₁ — 俯视图(从正上方往下看 X-Y平面)",
          "#1971c2","#e7f5ff")
    ox,oy=140,320   # 原点(底座)在图中的像素位置
    line([[ox,oy],[ox+330,oy]],"#495057",2,arrow_end=True); txt(ox+335,oy-8,"X",16,"#495057")
    line([[ox,oy],[ox,oy-240]],"#495057",2,arrow_end=True); txt(ox-20,oy-260,"Y",16,"#495057")
    dot(ox,oy,"#1971c2",5); txt(ox-58,oy+8,"底座O(原点)",12,"#1971c2")
    px,py=ox+210,oy-150
    line([[ox,oy],[px,py]],"#e8590c",3); dot(px,py,"#e03131",6)
    txt(px+8,py-16,"P(x,y) 目标俯视投影",12,"#e03131")
    arc(ox,oy,60,0,35.6,"#7048e8",2); txt(ox+66,oy-26,"θ₁",15,"#7048e8")
    txt(60,300+ -0,"",12)
    txt(58,364,"θ₁ = atan2(y, x)   ← 手臂在水平面\"对准\"目标的方向",13,"#1e1e1e",3)
    # 算例框
    rect(320,150,250,200,"#f08c00","#fff9db",2)
    txt(332,158,"数字算例",14,"#f08c00")
    txt(332,182,"目标 (x,y)=(0.3, 0.3) m\n\nθ₁=atan2(0.3,0.3)=45°\n\n★atan2 分象限:\n (0.3,-0.3)→ -45°\n (-0.3,-0.3)→ -135°\n 普通arctan都=45°会错!",12,"#1e1e1e",3)

    # ===== 面板2: 降维到竖直平面 =====
    panel(620,40,500,360,"② 降维: 底座转好后, 手臂锁进一个竖直平面",
          "#2f9e44","#ebfbee")
    bx,by=700,330
    line([[bx,by],[bx+360,by]],"#495057",2,arrow_end=True); txt(bx+365,by-8,"r",15,"#495057")
    txt(bx+300,by+10,"r=√(x²+y²) 水平距离",11,"#495057")
    line([[bx,by],[bx,by-250]],"#495057",2,arrow_end=True); txt(bx-18,by-270,"z",15,"#495057")
    dot(bx,by,"#2f9e44",5)
    tx,ty=bx+230,by-160
    dot(tx,ty,"#e03131",6); txt(tx+8,ty-14,"目标(r, z)",12,"#e03131")
    line([[bx,by],[tx,ty]],"#adb5bd",2,dash=True)
    rect(870,110,240,150,"#f08c00","#fff9db",2)
    txt(882,118,"数字算例(续)",14,"#f08c00")
    txt(882,142,"(0.3,0.3,0.2)m\nr=√(0.3²+0.3²)\n =√0.18=0.424 m\nz=0.2 m\n\n3D→2D: 5变量难题\n变成平面几何题",12,"#1e1e1e",3)

    _panel_cosine()
    _panel_multi_singular()
    _panel_nn()

def _panel_cosine():
    # ===== 面板3: 余弦定理求肘角 θ3 (核心) =====
    Y=440
    panel(40,Y,1080,430,"③ 核心! 余弦定理求肘角 θ₃ — 肩S/肘E/腕心W 构成三角形",
          "#e8590c","#fff4e6")
    # 三角形 S-E-W
    S=(150,Y+300); E=(320,Y+120); Wp=(500,Y+300)
    line([S,E],"#1971c2",4); line([E,Wp],"#0c8599",4); line([S,Wp],"#adb5bd",2,dash=True)
    dot(*S,"#e03131",6); dot(*E,"#e03131",6); dot(*Wp,"#e03131",6)
    txt(S[0]-42,S[1]+6,"S 肩",13,"#1971c2")
    txt(E[0]-8,E[1]-26,"E 肘",13,"#e8590c")
    txt(Wp[0]+8,Wp[1]+6,"W 腕心",13,"#0c8599")
    txt(215,Y+195,"L₁上臂",12,"#1971c2")
    txt(420,Y+195,"L₂前臂",12,"#0c8599")
    txt(300,Y+312,"D=√(r'²+z'²) 肩到腕心",12,"#868e96")
    arc(E[0],E[1],42,-63,-20,"#7048e8",2); txt(E[0]+2,E[1]+30,"∠E内角",11,"#7048e8")
    # 推导文字
    txt(600,Y+52,"余弦定理(勾股定理的推广, C=90°时退化为c²=a²+b²):",13,"#1e1e1e")
    txt(615,Y+78,"c² = a² + b² − 2ab·cos(C)",14,"#c2255c",3)
    txt(600,Y+112,"套到三角形(D对着肘角, L₁L₂是夹边):",13,"#1e1e1e")
    txt(615,Y+138,"D² = L₁² + L₂² − 2L₁L₂·cos(∠E)",14,"#1e1e1e",3)
    txt(600,Y+170,"★关键: 关节角θ₃ ≠ 内角∠E !",13,"#e03131")
    txt(600,Y+194,"伸直时 θ₃=0 但内角=180°, 故 ∠E = π − θ₃",12,"#1e1e1e")
    txt(600,Y+216,"又 cos(π−θ₃) = −cos(θ₃), 代入得:",12,"#1e1e1e")
    txt(600,Y+246,"cos θ₃ = (D²−L₁²−L₂²)/(2L₁L₂)",15,"#c2255c",3)
    txt(600,Y+276,"       = (r'²+z'²−L₁²−L₂²)/(2L₁L₂)",13,"#c2255c",3)
    # 算例
    rect(600,Y+306,500,110,"#f08c00","#fff9db",2)
    txt(612,Y+312,"数字算例  (L₁=L₂=0.15m, D=0.25m):",13,"#f08c00")
    txt(612,Y+338,"cosθ₃=(0.25²−0.15²−0.15²)/(2·0.15·0.15)\n"
                  "      =(0.0625−0.045)/0.045 =0.389\n"
                  "θ₃ = arccos(0.389) = ±67.1°   ← 为何是±? 见下方",12,"#1e1e1e",3)
    # 左下: 肩角θ2
    txt(60,Y+330,"肩角 θ₂ = β − ψ:",13,"#1e1e1e")
    txt(70,Y+354,"β=atan2(z',r') 直指腕心的仰角",11,"#1e1e1e")
    txt(70,Y+374,"ψ=atan2(L₂sinθ₃, L₁+L₂cosθ₃) 肘弯补偿",11,"#1e1e1e")
    txt(70,Y+394,"θ₂ = atan2(z',r') − atan2(L₂sinθ₃,L₁+L₂cosθ₃)",12,"#c2255c",3)

def _panel_multi_singular():
    # ===== 面板4: 多解(±) + 无解边界 =====
    Y=910
    panel(40,Y,540,360,"④ 为什么是 ±? 多解 & 无解的几何来源","#7048e8","#f3f0ff")
    # 肘朝上
    S=(130,Y+230); Wp=(330,Y+230); Eu=(230,Y+110); Ed=(230,Y+300)
    line([S,Eu],"#1971c2",3); line([Eu,Wp],"#0c8599",3)
    line([S,Ed],"#1971c2",3,dash=True); line([Ed,Wp],"#0c8599",3,dash=True)
    dot(*S,"#e03131",5); dot(*Wp,"#e03131",5); dot(*Eu,"#7048e8",5); dot(*Ed,"#f08c00",5)
    txt(S[0]-30,S[1]+6,"S",12,"#1971c2"); txt(Wp[0]+6,Wp[1]+6,"W",12,"#0c8599")
    txt(Eu[0]-6,Eu[1]-24,"E肘朝上 (+θ₃)",12,"#7048e8")
    txt(Ed[0]-6,Ed[1]+8,"E肘朝下 (−θ₃)",12,"#f08c00")
    txt(58,Y+320,"cos是偶函数 cos(θ)=cos(−θ) → arccos给出±两解",12,"#1e1e1e")
    txt(58,Y+340,"同一腕心, 肘上弯/下弯都到达 → 这就是IK\"多解\"",12,"#e03131")
    # 无解边界
    rect(320,Y+120,250,175,"#e03131","#fff5f5",2)
    txt(332,Y+128,"何时无解?",13,"#e03131")
    txt(332,Y+152,"arccos输入须∈[−1,1]:\n\n D > L₁+L₂ → 太远,\n   伸直也够不着\n D < |L₁−L₂| → 太近\n\n这两个边界围成\n环形\"工作空间\"",12,"#1e1e1e",3)

    # ===== 面板5: 奇异点 + 雅可比 + DLS =====
    panel(620,Y,500,360,"⑤ 奇异点 / 雅可比 / 阻尼(你问的概念)","#e03131","#fff5f5")
    # 伸直奇异示意
    a=(650,Y+120); b=(760,Y+120); c=(870,Y+120)
    line([a,b],"#1971c2",4); line([b,c],"#0c8599",4)
    dot(*a,"#e03131",5); dot(*b,"#e03131",5); dot(*c,"#e03131",5)
    txt(a[0]-14,a[1]+8,"S",11,"#1971c2"); txt(b[0]-4,b[1]+8,"E",11,"#e8590c"); txt(c[0]-4,c[1]+8,"W",11,"#0c8599")
    line([[c[0],c[1]],[c[0]+40,c[1]]],"#e03131",2,arrow_end=True)
    txt(c[0]+44,c[1]-8,"想再→前进?",10,"#e03131")
    txt(636,Y+150,"手臂伸直=边界奇异: W只能画弧(切向),",11,"#1e1e1e")
    txt(636,Y+168,"没法沿伸直方向前进→需无穷大关节速度!",11,"#e03131")
    txt(636,Y+196,"雅可比 J: 关节速度→末端速度的转换器",12,"#1e1e1e")
    txt(646,Y+216,"ẋ = J·θ̇ ,  Jᵢⱼ=∂xᵢ/∂θⱼ",13,"#1e1e1e",3)
    txt(636,Y+240,"奇异点: det(J)=0, J不可逆, J⁻¹里÷0→爆炸",11,"#e03131")
    txt(636,Y+266,"阻尼最小二乘救场(加λ²I防爆炸):",12,"#1e1e1e")
    txt(646,Y+288,"Δθ = Jᵀ(JJᵀ + λ²I)⁻¹ e",14,"#c2255c",3)
    txt(636,Y+314,"括号至少有λ², 永不接近0→不爆炸",11,"#1e1e1e")
    txt(636,Y+332,"代价:精度略降; λ自适应(近奇异w=√det(JJᵀ)→0时调大)",10,"#1e1e1e")

def _panel_nn():
    # ===== 面板6: 神经网络两个数学技巧 =====
    Y=1310
    panel(40,Y,1080,300,"⑥ 神经网络里的两个数学技巧","#0c8599","#e3fafc")
    # sin/cos 单位圆
    cx,cy=180,Y+150
    arc(cx,cy,70,0,360,"#0c8599",2)
    line([[cx-90,cy],[cx+90,cy]],"#adb5bd",1,arrow_end=True)
    line([[cx,cy+90],[cx,cy-90]],"#adb5bd",1,arrow_end=True)
    import math as _m
    for ang,col in [(1,"#e8590c"),(359,"#7048e8")]:
        rx=cx+70*_m.cos(_m.radians(ang)); ry=cy-70*_m.sin(_m.radians(ang))
        line([[cx,cy],[rx,ry]],col,2); dot(rx,ry,col,4)
    txt(cx-40,cy-115,"① 角度→sin/cos",13,"#0c8599")
    txt(70,Y+150,"周期性: 359°与1°\n只差2°但数值差358\n神经网络会当成\n天差地别!\n\n映射到单位圆:\n359°→(1.00,−0.017)\n 1° →(1.00,+0.017)\n两点几乎重合✓",11,"#1e1e1e",3)
    # tanh 限位
    txt(560,Y+50,"② tanh 保证不超关节限位(硬约束):",14,"#0c8599")
    txt(575,Y+82,"q = (q_hi+q_lo)/2 + (q_hi−q_lo)/2 · tanh(z)",14,"#c2255c",3)
    txt(560,Y+118,"tanh 值域永远 ∈ (−1, 1):",13,"#1e1e1e")
    txt(575,Y+144,"tanh=−1 → q = 中点−半幅 = q_lo (下限)",12,"#1e1e1e")
    txt(575,Y+166,"tanh=+1 → q = 中点+半幅 = q_hi (上限)",12,"#1e1e1e")
    txt(560,Y+194,"∵ tanh数学上不可能超出(−1,1)",12,"#1e1e1e")
    txt(560,Y+216,"∴ q 数学上不可能超出关节范围 → 硬约束",12,"#e03131")

if __name__=="__main__":
    build()
    save()
