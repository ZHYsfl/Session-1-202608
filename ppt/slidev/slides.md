---
theme: seriph
title: "Session 1 汇报 · 基础实验与 A2 进阶实战"
class: text-center
transition: slide-left
mdc: true
---

<style scoped>
.slidev-layout {
  background: linear-gradient(135deg, #22333d 0%, #3d5561 55%, #5a7a8a 100%) !important;
  width: 100% !important;
  height: 100% !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  align-items: center !important;
  color: white !important;
  text-shadow: 0 2px 10px rgba(0,0,0,0.35) !important;
}
.slidev-layout h1 {
  font-size: 3.6rem !important;
  font-weight: 500 !important;
  margin-bottom: 0.6rem !important;
  color: white !important;
}
.slidev-layout h2 {
  font-size: 1.35rem !important;
  font-weight: 400 !important;
  margin-bottom: 2.2rem !important;
  color: rgba(255,255,255,0.9) !important;
}
.slidev-layout .kicker {
  font-size: 0.95rem !important;
  letter-spacing: 0.35em !important;
  color: rgba(255,255,255,0.65) !important;
  margin-bottom: 1.2rem !important;
}
.slidev-layout .info {
  font-size: 0.95rem !important;
  line-height: 1.9 !important;
  color: rgba(255,255,255,0.85) !important;
}
</style>

<div class="kicker">SESSION 1 · 结课汇报</div>

# 让实验连成闭环

## 九个基础实验 → 具身智能系统 · A2 进阶实战

<div class="info">
第一部分 · 基础实验（汇报人：002）<br>
第二部分 · A2 进阶项目：VOA 强化学习防碰撞
</div>

---
layout: center
class: text-center
transition: slide-left
title: "汇报框架"
---

<h1 style="color: #5a7a8a; font-weight: 400; font-size: 2.5rem; margin-bottom: 3.5rem;">汇报框架</h1>

<div style="display: flex; justify-content: center; gap: 4rem;">

<div style="text-align: center; max-width: 340px;">
<div style="font-size: 2rem; font-weight: 500; color: #1e40af; margin-bottom: 0.4rem;">Part 01</div>
<div style="font-size: 1.3rem; font-weight: 500; color: #374151; margin-bottom: 0.5rem;">九个基础实验</div>
<div style="font-size: 0.95rem; color: #6b7280; line-height: 1.7;">感知 · 认知 · 学习 · 执行<br>如何连成一个机器人闭环<br><span style="color:#5a7a8a;">重点：实验⑨ 机械臂逆运动学</span></div>
</div>

<div style="text-align: center; max-width: 340px;">
<div style="font-size: 2rem; font-weight: 500; color: #166534; margin-bottom: 0.4rem;">Part 02</div>
<div style="font-size: 1.3rem; font-weight: 500; color: #374151; margin-bottom: 0.5rem;">A2 进阶实战</div>
<div style="font-size: 0.95rem; color: #6b7280; line-height: 1.7;">VOA 强化学习防碰撞线控底盘<br>三方协作架构 · 真机演示</div>
</div>

</div>

---
layout: center
class: text-center
transition: slide-left
title: "Part 01 · 基础实验"
---

<div style="font-size: 1.1rem; letter-spacing: 0.3em; color: #9db4be; margin-bottom: 1rem;">PART 01</div>

<div style="font-size: 2.8rem; font-weight: 400; color: #5a7a8a;">九个基础实验</div>

<div style="font-size: 1.15rem; color: #6b7280; margin-top: 0.8rem;">感知环境 · 建立空间认知 · 学习决策 · 执行动作</div>

<div style="font-size: 0.9rem; color: #9ca3af; margin-top: 2.5rem;">汇报人：002</div>

---
transition: slide-up
title: "总体框架：九个实验一个闭环"
---

<div style="font-size: 2.1rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.3rem;">九个实验共同回答：机器人怎样形成闭环？</div>

<div style="font-size: 0.9rem; color: #6b7280; margin-bottom: 1.6rem;">前一阶段的输出，就是后一阶段的输入</div>

<div style="display: flex; align-items: stretch; justify-content: center; gap: 0.6rem;">

<div class="p-4 bg-blue-50 rounded-lg text-center" style="flex: 1;">
<div style="font-size: 1.05rem; font-weight: 600; color: #1e40af; margin-bottom: 0.6rem;">环境感知</div>
<div style="font-size: 1.6rem; font-weight: 500; color: #1e40af;">① ⑦</div>
<div style="font-size: 0.8rem; color: #4b5563; margin-top: 0.5rem; line-height: 1.5;">目标检测与测距<br>道路分割</div>
</div>

<div style="align-self: center; color: #9ca3af; font-size: 1.3rem;">→</div>

<div class="p-4 bg-green-50 rounded-lg text-center" style="flex: 1;">
<div style="font-size: 1.05rem; font-weight: 600; color: #166534; margin-bottom: 0.6rem;">空间认知</div>
<div style="font-size: 1.6rem; font-weight: 500; color: #166534;">④ ⑥</div>
<div style="font-size: 0.8rem; color: #4b5563; margin-top: 0.5rem; line-height: 1.5;">RGB-D 三维重建<br>2D-SLAM 建图定位</div>
</div>

<div style="align-self: center; color: #9ca3af; font-size: 1.3rem;">→</div>

<div class="p-4 bg-orange-50 rounded-lg text-center" style="flex: 1;">
<div style="font-size: 1.05rem; font-weight: 600; color: #c2410c; margin-bottom: 0.6rem;">预测与学习</div>
<div style="font-size: 1.6rem; font-weight: 500; color: #c2410c;">② ③ ⑤ ⑧</div>
<div style="font-size: 0.8rem; color: #4b5563; margin-top: 0.5rem; line-height: 1.5;">碰撞预测 · 强化学习<br>模仿学习 · 端到端驾驶</div>
</div>

<div style="align-self: center; color: #9ca3af; font-size: 1.3rem;">→</div>

<div class="p-4 bg-purple-50 rounded-lg text-center" style="flex: 1;">
<div style="font-size: 1.05rem; font-weight: 600; color: #6d28d9; margin-bottom: 0.6rem;">动作执行</div>
<div style="font-size: 1.6rem; font-weight: 500; color: #6d28d9;">⑨</div>
<div style="font-size: 0.8rem; color: #4b5563; margin-top: 0.5rem; line-height: 1.5;">机械臂逆运动学<br>关节控制</div>
</div>

</div>

<div class="mt-8 mx-auto text-center" style="font-size: 0.9rem; color: #4b5563;">
环境反馈重新进入传感器，<span style="color: #5a7a8a; font-weight: 600;">闭环再次开始</span>
</div>

---
transition: slide-up
title: "实验① 目标检测与测距"
---

<div style="font-size: 1.9rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.25rem;">基础实验 ① ｜ 检测回答“有什么”，测距回答“多远”</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 1rem;">YOLO 定位车辆与行人，再用针孔模型估计距离并分级预警 · <span style="color:#1e40af;">环境感知</span></div>

<div style="display: flex; gap: 1.6rem; align-items: flex-start;">

<div style="flex: 1.05;" class="text-left">

<div class="p-3 bg-blue-50 rounded-lg mb-2">
<div style="font-size: 0.85rem; font-weight: 600; color: #1e40af;">输入</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.45;">单目相机图像 / 视频 / 实时画面；YOLO 输出目标框</div>
</div>

<div class="p-3 bg-blue-50 rounded-lg mb-2">
<div style="font-size: 0.85rem; font-weight: 600; color: #1e40af;">原理</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.45;">针孔模型 <code>d = H·f / h</code>：目标真实尺寸、焦距与像素尺寸共同决定距离</div>
</div>

<div class="p-3 bg-blue-50 rounded-lg">
<div style="font-size: 0.85rem; font-weight: 600; color: #1e40af;">工程</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.45;">连续多帧平滑 + 宽高双测距融合；小于 3 m 危险、3–5 m 警告、大于 5 m 安全</div>
</div>

</div>

<div style="flex: 1;" class="text-center">
<video src="./assets/videos/B1.mp4" autoplay muted loop playsinline controls style="width: 100%; max-height: 275px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px; background: #000;"></video>
<div style="font-size: 1.5rem; font-weight: 500; color: #1e40af; margin-top: 0.6rem;">3 / 5 m</div>
<div style="font-size: 0.8rem; color: #6b7280;">两级距离告警阈值</div>
</div>

</div>

---
transition: slide-up
title: "实验⑦ 道路分割"
---

<div style="font-size: 1.9rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.25rem;">基础实验 ⑦ ｜ 像素级道路掩膜把“能否走”交给视觉模型</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 1rem;">实例分割从目标框进一步细化到道路的像素级边界 · <span style="color:#1e40af;">环境感知</span></div>

<div style="display: flex; gap: 1.2rem; align-items: stretch;">

<div style="flex: 2.2;" class="grid grid-cols-3 gap-3 text-left">

<div class="p-4 bg-blue-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #1e40af; margin-bottom: 0.4rem;">数据</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">100 张多边形标注图像，按 80 / 10 / 10 划分训练、验证与测试</div>
</div>

<div class="p-4 bg-green-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #166534; margin-bottom: 0.4rem;">方法</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">冻结前 10 层进行迁移学习，在小数据集上降低过拟合</div>
</div>

<div class="p-4 bg-orange-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #c2410c; margin-bottom: 0.4rem;">结果</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">Mask mAP@0.5 = <b>0.995</b>；更严格的 mAP@0.5:0.95 = <b>0.879</b></div>
</div>

</div>

<div style="flex: 0.9;" class="p-4 rounded-lg border border-slate-200 bg-white text-center flex flex-col justify-center">
<div style="font-size: 2rem; font-weight: 500; color: #1e40af;">26.72 ms</div>
<div style="font-size: 0.82rem; color: #6b7280; margin-top: 0.4rem; line-height: 1.5;">单图总延迟<br>约 37 FPS，满足实时</div>
</div>

</div>

---
transition: slide-up
title: "实验④ RGB-D 三维重建"
---

<div style="font-size: 1.9rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.25rem;">基础实验 ④ ｜ RGB-D 把二维像素恢复为可测量的三维场景</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 1rem;">深度解除投影歧义，多视角 TSDF 融合得到一致表面 · <span style="color:#166534;">空间认知</span></div>

<div style="display: flex; gap: 1.6rem; align-items: flex-start;">

<div style="flex: 1.05;" class="text-left">

<div class="p-3 bg-green-50 rounded-lg mb-2">
<div style="font-size: 0.85rem; font-weight: 600; color: #166534;">输入</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.45;">彩色图、逐像素深度与相机位姿（Webots 仿真 RGB-D 相机）</div>
</div>

<div class="p-3 bg-green-50 rounded-lg mb-2">
<div style="font-size: 0.85rem; font-weight: 600; color: #166534;">过程</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.45;">像素反投影为点云 → 多帧对齐 → TSDF 融合 → Marching Cubes 提取网格</div>
</div>

<div class="p-3 bg-green-50 rounded-lg">
<div style="font-size: 0.85rem; font-weight: 600; color: #166534;">验证</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.45;">干净仿真数据中地面 RMS 约 2.7 mm；5 mm 体素兼顾质量与速度</div>
</div>

</div>

<div style="flex: 1;" class="text-center">
<video src="./assets/videos/B4.mp4" autoplay muted loop playsinline controls style="width: 100%; max-height: 275px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px; background: #000;"></video>
<div style="font-size: 1.5rem; font-weight: 500; color: #166534; margin-top: 0.6rem;">4.10 mm</div>
<div style="font-size: 0.8rem; color: #6b7280;">10 mm 深度噪声下重建误差；直接拼接为 10.36 mm</div>
</div>

</div>

---
transition: slide-up
title: "实验⑥ 2D-SLAM"
---

<div style="font-size: 1.9rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.25rem;">基础实验 ⑥ ｜ SLAM 同时估计机器人位姿与环境地图</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 1rem;">激光扫描负责匹配环境，里程计提供运动先验 · <span style="color:#166534;">空间认知</span></div>

<div style="display: flex; gap: 1.2rem; align-items: stretch;">

<div style="flex: 2.2;" class="grid grid-cols-3 gap-3 text-left">

<div class="p-4 bg-blue-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #1e40af; margin-bottom: 0.4rem;">数据</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">办公室 756 帧、实验室 641 帧；每帧 180 束 360° 激光</div>
</div>

<div class="p-4 bg-green-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #166534; margin-bottom: 0.4rem;">方法</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">RMHC-SLAM 维护位姿假设，并持续更新占据栅格地图；对比纯激光与里程计融合</div>
</div>

<div class="p-4 bg-orange-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #c2410c; margin-bottom: 0.4rem;">结果</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">融合里程计后覆盖面积提升 4.7% 与 9.4%，速度超过 270 帧 / 秒</div>
</div>

</div>

<div style="flex: 0.9;" class="p-4 rounded-lg border border-slate-200 bg-white text-center flex flex-col justify-center">
<div style="font-size: 2rem; font-weight: 500; color: #166534;">+9.4%</div>
<div style="font-size: 0.82rem; color: #6b7280; margin-top: 0.4rem; line-height: 1.5;">复杂实验室场景的<br>覆盖面积增益</div>
</div>

</div>

---
transition: slide-up
title: "实验② 碰撞预测"
---

<div style="font-size: 1.9rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.25rem;">基础实验 ② ｜ 多方向测距把瞬时距离变成未来碰撞概率</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 1rem;">九向传感器与运动状态共同预测未来 30 帧是否碰撞 · <span style="color:#c2410c;">预测与学习</span></div>

<div style="display: flex; gap: 1.6rem; align-items: flex-start;">

<div style="flex: 1.05;" class="text-left">

<div class="p-3 bg-orange-50 rounded-lg mb-2">
<div style="font-size: 0.85rem; font-weight: 600; color: #c2410c;">输入</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.45;">9 个方向距离 + 线速度 / 角速度，共 11 维特征</div>
</div>

<div class="p-3 bg-orange-50 rounded-lg mb-2">
<div style="font-size: 0.85rem; font-weight: 600; color: #c2410c;">模型</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.45;">MLP 二分类；标签表示未来 30 帧内是否发生碰撞</div>
</div>

<div class="p-3 bg-orange-50 rounded-lg">
<div style="font-size: 0.85rem; font-weight: 600; color: #c2410c;">扩展</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.45;">DQN 决定何时避障；V3 在 500 个未见回合中碰撞率仅 1.20%</div>
</div>

</div>

<div style="flex: 1;" class="text-center">
<video src="./assets/videos/B2.mp4" autoplay muted loop playsinline controls style="width: 100%; max-height: 275px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px; background: #000;"></video>
<div style="font-size: 1.5rem; font-weight: 500; color: #c2410c; margin-top: 0.6rem;">83.65%</div>
<div style="font-size: 0.8rem; color: #6b7280;">测试准确率；召回率 84.08%</div>
</div>

</div>

---
transition: slide-up
title: "实验③ CartPole 强化学习"
---

<div style="font-size: 1.9rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.25rem;">基础实验 ③ ｜ 强化学习通过奖励试错学会保持平衡</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 1rem;">状态、动作与奖励把控制问题转化为策略学习 · <span style="color:#c2410c;">预测与学习</span></div>

<div style="display: flex; gap: 1.6rem; align-items: flex-start;">

<div style="flex: 1.05;" class="text-left">

<div class="p-3 bg-orange-50 rounded-lg mb-2">
<div style="font-size: 0.85rem; font-weight: 600; color: #c2410c;">状态与动作</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.45;">小车位置 / 速度、杆角度 / 角速度；动作是向左或向右</div>
</div>

<div class="p-3 bg-orange-50 rounded-lg mb-2">
<div style="font-size: 0.85rem; font-weight: 600; color: #c2410c;">比较</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.45;">3 个训练种子；Q-Learning 与多种 DQN 配置均做封存测试</div>
</div>

<div class="p-3 bg-orange-50 rounded-lg">
<div style="font-size: 0.85rem; font-weight: 600; color: #c2410c;">发现</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.45;">归一化是 DQN 最有效改动；状态分箱并非越细越好</div>
</div>

</div>

<div style="flex: 1;" class="text-center">
<video src="./assets/videos/B3.mp4" autoplay muted loop playsinline controls style="width: 100%; max-height: 275px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px; background: #000;"></video>
<div style="font-size: 1.5rem; font-weight: 500; color: #c2410c; margin-top: 0.6rem;">91.67%</div>
<div style="font-size: 0.8rem; color: #6b7280;">Q-Learning Medium 成功率；平均 490.30 步</div>
</div>

</div>

---
transition: slide-up
title: "实验⑤ 模仿学习"
---

<div style="font-size: 1.9rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.25rem;">基础实验 ⑤ ｜ 模仿学习从专家示范开始，但必须面对分布偏移</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 1rem;">行为克隆学习专家动作，DAgger 再收集策略真正访问的状态 · <span style="color:#c2410c;">预测与学习</span></div>

<div style="display: flex; gap: 1.2rem; align-items: stretch;">

<div style="flex: 2.2;" class="grid grid-cols-3 gap-3 text-left">

<div class="p-4 bg-blue-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #1e40af; margin-bottom: 0.4rem;">BC 行为克隆</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">把“状态 → 专家动作”的映射当作监督学习直接拟合</div>
</div>

<div class="p-4 bg-orange-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #c2410c; margin-bottom: 0.4rem;">问题：分布偏移</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">策略一旦犯错，就会进入专家数据没有覆盖的新状态，误差不断累积</div>
</div>

<div class="p-4 bg-green-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #166534; margin-bottom: 0.4rem;">DAgger</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">滚动执行当前策略，请专家标注新状态并迭代聚合数据</div>
</div>

</div>

<div style="flex: 0.9;" class="p-4 rounded-lg border border-slate-200 bg-white text-center flex flex-col justify-center">
<div style="font-size: 1.7rem; font-weight: 500; color: #c2410c;">20% → 100%</div>
<div style="font-size: 0.82rem; color: #6b7280; margin-top: 0.4rem; line-height: 1.5;">CartPole 闭环成功率<br>80 个评估回合</div>
</div>

</div>

---
transition: slide-up
title: "实验⑧ 端到端驾驶"
---

<div style="font-size: 1.9rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.25rem;">基础实验 ⑧ ｜ 端到端驾驶把相机图像直接映射为转向控制</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 1rem;">CNN 不显式拆分车道检测与规划，而是直接回归方向盘角度 · <span style="color:#c2410c;">预测与学习</span></div>

<div style="display: flex; gap: 1.2rem; align-items: stretch;">

<div style="flex: 2.2;" class="grid grid-cols-3 gap-3 text-left">

<div class="p-4 bg-blue-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #1e40af; margin-bottom: 0.4rem;">数据</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">7698 条三相机记录；6158 训练、1540 验证</div>
</div>

<div class="p-4 bg-green-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #166534; margin-bottom: 0.4rem;">网络</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">32×128 图像输入，卷积特征后回归连续转角，共 972,225 个参数</div>
</div>

<div class="p-4 bg-orange-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #c2410c; margin-bottom: 0.4rem;">边界</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">已验证模型与损失记录；原始数据不在仓库，闭环成功率未报告</div>
</div>

</div>

<div style="flex: 0.9;" class="p-4 rounded-lg border border-slate-200 bg-white text-center flex flex-col justify-center">
<div style="font-size: 2rem; font-weight: 500; color: #1e40af;">0.0144</div>
<div style="font-size: 0.82rem; color: #6b7280; margin-top: 0.4rem; line-height: 1.5;">最终验证 MSE<br>训练 30 轮</div>
</div>

</div>

---
transition: slide-left
title: "实验⑨ 任务定义"
---

<div style="font-size: 1.9rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.25rem;">实验⑨ · 重点 ｜ 让 SO-101 到达指定末端位姿</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 1rem;">任务不仅是求出关节角，还要让参数不准的低成本机械臂真正到位 · <span style="color:#6d28d9;">动作执行</span></div>

<div style="display: flex; gap: 1.2rem; align-items: stretch;">

<div style="flex: 2.2;" class="grid grid-cols-3 gap-3 text-left">

<div class="p-4 bg-purple-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #6d28d9; margin-bottom: 0.4rem;">对象</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">SO-101：6 个舵机，前 5 关节参与 IK，末端带 98 mm 把手</div>
</div>

<div class="p-4 bg-blue-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #1e40af; margin-bottom: 0.4rem;">数据</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">真机采集 600 组关节角，形成覆盖安全工作空间的数据集</div>
</div>

<div class="p-4 bg-green-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #166534; margin-bottom: 0.4rem;">目标</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">比较解析、数值、神经、混合四类 IK，并在真机闭环验证</div>
</div>

</div>

<div style="flex: 0.9;" class="p-4 rounded-lg border border-slate-200 bg-white text-center flex flex-col justify-center">
<div style="font-size: 2rem; font-weight: 500; color: #6d28d9;">5-DoF</div>
<div style="font-size: 0.82rem; color: #6b7280; margin-top: 0.4rem; line-height: 1.5;">任务空间 (x, y, z, α, ψ)<br>→ 关节角 q₁…q₅</div>
</div>

</div>

---
transition: slide-up
title: "实验⑨ 方法链"
---

<div style="font-size: 1.9rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.25rem;">实验⑨ 方法链：先求解，再用真实反馈消除偏差</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 1.1rem;">逆运动学求解 → 真机反馈补偿</div>

<div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem; flex-wrap: nowrap;">

<div class="p-3 bg-purple-50 rounded-lg text-center" style="min-width: 108px;">
<div style="font-size: 0.82rem; font-weight: 600; color: #6d28d9;">目标位姿</div>
<div style="font-size: 0.75rem; color: #4b5563; margin-top: 0.25rem;">(x, y, z, α, ψ)</div>
</div>
<div style="color: #9ca3af;">→</div>

<div class="p-3 bg-blue-50 rounded-lg text-center" style="min-width: 128px;">
<div style="font-size: 0.82rem; font-weight: 600; color: #1e40af;">四类求解器</div>
<div style="font-size: 0.75rem; color: #4b5563; margin-top: 0.25rem;">解析 / 数值 / NN / 混合</div>
</div>
<div style="color: #9ca3af;">→</div>

<div class="p-3 bg-green-50 rounded-lg text-center" style="min-width: 128px;">
<div style="font-size: 0.82rem; font-weight: 600; color: #166534;">安全约束</div>
<div style="font-size: 0.75rem; color: #4b5563; margin-top: 0.25rem;">tanh 限位 + 路径检查</div>
</div>
<div style="color: #9ca3af;">→</div>

<div class="p-3 bg-orange-50 rounded-lg text-center" style="min-width: 108px;">
<div style="font-size: 0.82rem; font-weight: 600; color: #c2410c;">开环命令</div>
<div style="font-size: 0.75rem; color: #4b5563; margin-top: 0.25rem;">发送 q_target</div>
</div>
<div style="color: #9ca3af;">→</div>

<div class="p-3 bg-slate-100 rounded-lg text-center" style="min-width: 108px;">
<div style="font-size: 0.82rem; font-weight: 600; color: #334155;">编码器</div>
<div style="font-size: 0.75rem; color: #4b5563; margin-top: 0.25rem;">读取 q_actual</div>
</div>
<div style="color: #9ca3af;">→</div>

<div class="p-3 bg-purple-50 rounded-lg text-center" style="min-width: 128px;">
<div style="font-size: 0.82rem; font-weight: 600; color: #6d28d9;">残差更新</div>
<div style="font-size: 0.75rem; color: #4b5563; margin-top: 0.25rem;">q_cmd += 0.8·Δq，迭代至收敛</div>
</div>

</div>

<div class="rounded-xl border border-slate-300 bg-white px-4 py-3 text-center mt-8 mx-auto" style="max-width: 720px;">
<span style="font-size: 0.85rem; font-weight: 600; color: #334155;">核心公式：</span>
<code style="font-size: 0.95rem; color: #6d28d9;">q_cmd ← q_cmd + 0.8 ( q_target − q_actual )</code>
<div style="font-size: 0.78rem; color: #6b7280; margin-top: 0.3rem;">循环至关节残差进入阈值 —— 平均误差 33.56 mm → 3.96 mm</div>
</div>

---
transition: slide-up
title: "实验⑨ 四种 IK 对比"
---

<div style="font-size: 1.9rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.25rem;">实验⑨ ｜ 四种 IK 各有精度与速度取舍</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 1rem;">同一批 50 个目标点、远初值测试：精度、速度和参数依赖形成明确取舍</div>

<div class="max-w-4xl mx-auto text-left" style="font-size: 0.86rem;">

| 方法 | 成功率 | 位置误差 | 单次求解耗时 | 特点 |
|------|--------|----------|--------------|------|
| **解析** | 4 / 50 | 243.5 mm | **0.015 ms** | 最快，但参数不准就失效 |
| **数值** | **49 / 50** | **0.89 mm** | 28.35 ms | 精度最高，但计算慢 |
| **神经网络** | — | 17.24 mm | 0.69 ms | 快且稳，精度中等 |
| **混合（3 步）** | — | 14.59 mm | 5.44 ms | NN 初值 + 数值精修，可调步数 |

</div>

<div class="mt-6 text-center">
<span style="font-size: 1.6rem; font-weight: 500; color: #6d28d9;">≈ 1,800×</span>
<span style="font-size: 0.85rem; color: #6b7280; margin-left: 0.6rem;">解析法与数值法的求解时间差；混合步数可按精度-速度需求调节</span>
</div>

---
transition: slide-up
title: "实验⑨ 真机误差地板"
---

<div style="font-size: 1.9rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.25rem;">实验⑨ ｜ 真机开环卡在 36 mm 误差地板</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 1rem;">同一批 5 个目标点统一验证：软件差异明显，进入真机后却落在同一误差地板</div>

<div style="display: flex; gap: 1.2rem; align-items: stretch;">

<div style="flex: 2.2;" class="grid grid-cols-3 gap-3 text-left">

<div class="p-4 bg-blue-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #1e40af; margin-bottom: 0.4rem;">数值</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">软件 <b>0.90 mm</b><br>→ 真机 <b>35.59 mm</b></div>
</div>

<div class="p-4 bg-green-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #166534; margin-bottom: 0.4rem;">神经</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">软件 <b>15.15 mm</b><br>→ 真机 <b>35.94 mm</b></div>
</div>

<div class="p-4 bg-orange-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #c2410c; margin-bottom: 0.4rem;">混合</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">软件 <b>5.09 mm</b><br>→ 真机 <b>36.04 mm</b></div>
</div>

</div>

<div style="flex: 0.9;" class="p-4 rounded-lg border border-slate-200 bg-white text-center flex flex-col justify-center">
<div style="font-size: 2rem; font-weight: 500; color: #c2410c;">≈ 36 mm</div>
<div style="font-size: 0.82rem; color: #6b7280; margin-top: 0.4rem; line-height: 1.5;">重力下垂、齿隙与 3D 打印<br>公差主导真机误差</div>
</div>

</div>

<div class="mt-6 text-center" style="font-size: 0.9rem; color: #4b5563;">
瓶颈不在算法，而在机械 —— <span style="color: #5a7a8a; font-weight: 600;">这正是需要反馈补偿的原因</span>
</div>

---
transition: slide-up
title: "实验⑨ 关节闭环"
---

<div style="font-size: 1.9rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.25rem;">实验⑨ ｜ 关节闭环让平均误差下降 88%</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 1rem;">不用相机，只用舵机编码器读回实际关节角，并逐轮补偿命令残差</div>

<div style="display: flex; gap: 1.2rem; align-items: stretch;">

<div style="flex: 2.2;" class="grid grid-cols-3 gap-3 text-left">

<div class="p-4 bg-purple-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #6d28d9; margin-bottom: 0.4rem;">目标</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">IK 给出 q_target，执行后编码器读取 q_actual</div>
</div>

<div class="p-4 bg-blue-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #1e40af; margin-bottom: 0.4rem;">更新</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;"><code>q_cmd ← q_cmd + 0.8 (q_target − q_actual)</code></div>
</div>

<div class="p-4 bg-green-50 rounded-lg">
<div style="font-size: 0.9rem; font-weight: 600; color: #166534; margin-bottom: 0.4rem;">收敛</div>
<div style="font-size: 0.84rem; color: #374151; line-height: 1.55;">5 个测试点最多 5 轮；最终误差 1.0 – 9.9 mm</div>
</div>

</div>

<div style="flex: 0.9;" class="p-4 rounded-lg border border-slate-200 bg-white text-center flex flex-col justify-center">
<div style="font-size: 2rem; font-weight: 500; color: #166534;">88%</div>
<div style="font-size: 0.82rem; color: #6b7280; margin-top: 0.4rem; line-height: 1.5;">平均位置误差下降<br>5 个测试点全部改善</div>
</div>

</div>

---
transition: slide-up
title: "实验⑨ 成果演示"
---

<div style="font-size: 1.9rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.25rem;">实验⑨ ｜ 成果演示：精准推落物块，瓶体保持稳定</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 1rem;">机械臂从接触点 A 沿安全轨迹抬升到 B，只推物块并保持瓶体稳定</div>

<div style="display: flex; gap: 1.6rem; align-items: flex-start;">

<div style="flex: 1.05;" class="text-left">

<div class="p-3 bg-purple-50 rounded-lg mb-2">
<div style="font-size: 0.85rem; font-weight: 600; color: #6d28d9;">示教</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.45;">手动记录 28 个轨迹点，平滑后生成连续关节轨迹</div>
</div>

<div class="p-3 bg-purple-50 rounded-lg mb-2">
<div style="font-size: 0.85rem; font-weight: 600; color: #6d28d9;">安全</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.45;">全路径检查真实把手高度，Z ≥ 192 mm 才允许执行</div>
</div>

<div class="p-3 bg-purple-50 rounded-lg">
<div style="font-size: 0.85rem; font-weight: 600; color: #6d28d9;">效果</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.45;">物块被推落、瓶体保持稳定，验证到位精度与轨迹精度</div>
</div>

</div>

<div style="flex: 1;" class="text-center">
<video src="./assets/videos/B9.mp4" autoplay muted loop playsinline controls style="width: 100%; max-height: 275px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px; background: #000;"></video>
<div style="font-size: 1.5rem; font-weight: 500; color: #6d28d9; margin-top: 0.6rem;">4.0 mm</div>
<div style="font-size: 0.8rem; color: #6b7280;">闭环平均误差，支撑接触式操作任务</div>
</div>

</div>

---
transition: slide-up
title: "综合联系：两条闭环"
---

<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.3rem;">实验之间的关系：落在两条共享感知基础的闭环</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 1.2rem;">移动机器人 / 自动驾驶 · 机械臂 / 操作</div>

<div style="display: flex; gap: 1.2rem; justify-content: center;">

<div class="p-4 bg-blue-50 rounded-lg text-left" style="flex: 1; max-width: 420px;">
<div style="font-size: 0.95rem; font-weight: 600; color: #1e40af; margin-bottom: 0.5rem; text-align: center;">移动机器人 / 自动驾驶</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.8;">
<b>感知</b>　① 测距 + ⑦ 道路分割<br>
<b>风险</b>　② 碰撞预测<br>
<b>策略</b>　③ 强化学习 + ⑤ 模仿学习<br>
<b>驾驶</b>　⑧ 图像直接到转向
</div>
</div>

<div class="p-4 bg-purple-50 rounded-lg text-left" style="flex: 1; max-width: 420px;">
<div style="font-size: 0.95rem; font-weight: 600; color: #6d28d9; margin-bottom: 0.5rem; text-align: center;">机械臂 / 操作</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.8;">
<b>场景</b>　④ RGB-D 三维重建<br>
<b>定位</b>　⑥ SLAM<br>
<b>执行</b>　⑨ 逆运动学与关节控制<br>
<b>反馈</b>　编码器闭环补偿
</div>
</div>

</div>

<div class="rounded-xl border border-slate-300 bg-white px-4 py-3 text-center mt-6 mx-auto" style="max-width: 780px; font-size: 0.88rem; color: #334155;">
<span style="font-weight: 600;">共同机制：</span>传感器产生状态 → 算法形成判断 → 控制器执行动作 → 环境反馈再次进入传感器
</div>

---
transition: slide-up
title: "结论"
---

<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.3rem;">结论：智能来自闭环，而不是某一个模型</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 1.2rem;">看见环境 · 理解与决策 · 动作与反馈</div>

<div class="grid grid-cols-3 gap-4 max-w-4xl mx-auto text-left">

<div class="p-4 bg-blue-50 rounded-lg">
<div style="font-size: 0.95rem; font-weight: 600; color: #1e40af; margin-bottom: 0.4rem; text-align: center;">看见环境</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.6; text-align: center;">检测 · 分割 · 重建 · 建图</div>
</div>

<div class="p-4 bg-orange-50 rounded-lg">
<div style="font-size: 0.95rem; font-weight: 600; color: #c2410c; margin-bottom: 0.4rem; text-align: center;">理解与决策</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.6; text-align: center;">预测 · 强化学习 · 模仿学习</div>
</div>

<div class="p-4 bg-green-50 rounded-lg">
<div style="font-size: 0.95rem; font-weight: 600; color: #166534; margin-bottom: 0.4rem; text-align: center;">动作与反馈</div>
<div style="font-size: 0.82rem; color: #374151; line-height: 1.6; text-align: center;">车辆控制 · 逆运动学 · 闭环补偿</div>
</div>

</div>

<div class="mt-8 text-center" style="font-size: 0.95rem; color: #4b5563;">
下一步：把分散实验接入<span style="color: #5a7a8a; font-weight: 600;">同一机器人系统</span>，并重点验证真实环境泛化 —— 这正是第二部分的 A2
</div>

---
transition: slide-up
title: "深入阅读：论文报告"
---

<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.3rem;">想探索更多细节？请研读我们的论文报告</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 0.8rem;">每个实验一篇 LaTeX 英文 Paper（≥ 4 页），含消融与指标分析 · <code>paper/&lt;工号&gt;/</code></div>

<div class="grid grid-cols-4 gap-x-4 gap-y-2 max-w-5xl mx-auto">

<div class="text-center">
<img src="./assets/papers/paper-009.png" style="height: 148px; width: 100%; object-fit: cover; object-position: top; border: 1px solid #e5e7eb; border-radius: 8px;" />
<div style="font-size: 0.75rem; color: #4b5563; margin-top: 0.3rem;">009 · B1 单目测距</div>
</div>

<div class="text-center">
<img src="./assets/papers/paper-007.png" style="height: 148px; width: 100%; object-fit: cover; object-position: top; border: 1px solid #e5e7eb; border-radius: 8px;" />
<div style="font-size: 0.75rem; color: #4b5563; margin-top: 0.3rem;">007 · B2 碰撞预测</div>
</div>

<div class="text-center">
<img src="./assets/papers/paper-006.png" style="height: 148px; width: 100%; object-fit: cover; object-position: top; border: 1px solid #e5e7eb; border-radius: 8px;" />
<div style="font-size: 0.75rem; color: #4b5563; margin-top: 0.3rem;">006 · B3 CartPole</div>
</div>

<div class="text-center">
<img src="./assets/papers/paper-001.png" style="height: 148px; width: 100%; object-fit: cover; object-position: top; border: 1px solid #e5e7eb; border-radius: 8px;" />
<div style="font-size: 0.75rem; color: #4b5563; margin-top: 0.3rem;">001 · B4 三维重建</div>
</div>

<div class="text-center">
<img src="./assets/papers/paper-004.png" style="height: 148px; width: 100%; object-fit: cover; object-position: top; border: 1px solid #e5e7eb; border-radius: 8px;" />
<div style="font-size: 0.75rem; color: #4b5563; margin-top: 0.3rem;">004 · B5 模仿学习</div>
</div>

<div class="text-center">
<img src="./assets/papers/paper-005.png" style="height: 148px; width: 100%; object-fit: cover; object-position: top; border: 1px solid #e5e7eb; border-radius: 8px;" />
<div style="font-size: 0.75rem; color: #4b5563; margin-top: 0.3rem;">005 · B6 2D-SLAM</div>
</div>

<div class="text-center">
<img src="./assets/papers/paper-003.png" style="height: 148px; width: 100%; object-fit: cover; object-position: top; border: 1px solid #e5e7eb; border-radius: 8px;" />
<div style="font-size: 0.75rem; color: #4b5563; margin-top: 0.3rem;">003 · B7 道路分割</div>
</div>

<div class="text-center">
<img src="./assets/papers/paper-002.png" style="height: 148px; width: 100%; object-fit: cover; object-position: top; border: 1px solid #e5e7eb; border-radius: 8px;" />
<div style="font-size: 0.75rem; color: #4b5563; margin-top: 0.3rem;">002 · B9 机械臂 IK</div>
</div>

</div>

---
layout: center
class: text-center
transition: slide-left
title: "Part 02 · A2 进阶实战"
---

<div style="font-size: 1.1rem; letter-spacing: 0.3em; color: #9db4be; margin-bottom: 1rem;">PART 02</div>

<div style="font-size: 2.8rem; font-weight: 400; color: #5a7a8a;">A2 进阶实战</div>

<div style="font-size: 1.15rem; color: #6b7280; margin-top: 0.8rem;">VOA（Vision-Other-Action）强化学习防碰撞线控底盘</div>

<div style="font-size: 0.9rem; color: #9ca3af; margin-top: 2.5rem;">001 硬件侧 · 003 算法侧 · 009 模型侧</div>

---
transition: slide-up
title: "A2 系统架构"
---

<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.3rem;">A2 ｜ 三方协作架构：仿真与真机同一协议</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 0.6rem;">001 硬件侧（Webots 仿真 / 真机 env server）⇄ WebSocket ⇄ 003 算法侧（SAC client）⇄ 进程内 import ⇄ 009 模型侧（网络定义）· 协议 v1.1 唯一权威定义</div>

<div class="w-full flex justify-center">
<img src="./assets/pics/A2-arch.png" alt="A2 架构图" style="max-height: 430px; width: auto; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px;" />
</div>

---
transition: slide-up
title: "A2 演示"
---

<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.3rem;">A2 ｜ 演示：从仿真训练到真机运行</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 0.6rem;">SAC 策略驱动线控底盘绕障到点；真机阶段支持人工摆车与遥控介入（human-in-the-loop）</div>

<div class="w-full flex justify-center">
<div style="width: 760px; max-width: 92%;">
<video src="./assets/videos/A2.mp4" autoplay muted loop playsinline controls style="width: 100%; max-height: 430px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px; background: #000;"></video>
</div>
</div>

---
transition: slide-up
title: "团队协作"
---

<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.3rem;">团队协作：分支开发 · PR 评审 · 合并主线</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 0.8rem;">main + 001–010 共 11 个分支 · main 禁止强制 push · 每个 PR 至少一名他人审核通过后方可合并</div>

<div style="display: flex; gap: 1rem; justify-content: center; align-items: flex-start;">

<div class="text-center" style="flex: 1;">
<img src="./assets/pics/github-1.png" alt="GitHub PR 协作记录 1" style="width: 100%; max-height: 390px; object-fit: cover; object-position: top; border: 1px solid #e5e7eb; border-radius: 10px;" />
</div>

<div class="text-center" style="flex: 1;">
<img src="./assets/pics/github-2.png" alt="GitHub PR 协作记录 2" style="width: 100%; max-height: 390px; object-fit: cover; object-position: top; border: 1px solid #e5e7eb; border-radius: 10px;" />
</div>

</div>

---
transition: fade
class: text-center
title: "谢谢聆听"
---

# 谢谢聆听

<div class="mt-6 text-lg">

九个基础实验连成闭环，A2 把闭环送上真机

</div>

<div class="mt-8 text-base opacity-70">

恳请各位批评指正 · Q & A

</div>
