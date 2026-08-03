# Session 1 汇报 PPT（Slidev）

主题：九个基础实验连成机器人闭环（实验⑨ 机械臂重点）+ A2 进阶实战（VOA 强化学习防碰撞）。

## 放映（推荐）

```bash
cd /home/zane/session_1/ppt/slidev
npm run dev        # 打开 http://localhost:3030 ，全屏即放映
```

- 方向键 / 空格翻页；`o` 总览；`f` 全屏。
- 视频页（B1/B2/B3/B4/B9、A2）进入后自动静音循环播放，点击视频可手动控制。

## 构建静态站点

```bash
npm run build      # 输出到 dist/，任意静态服务器托管即可
```

## 导出 PDF（可选）

需要 Playwright 浏览器。本机缺 `libnspr4/libnss3/libasound2` 三个系统库，
已把对应 deb 解压在 `../.debs/root`（未改动系统），导出时这样注入：

```bash
LD_LIBRARY_PATH=/home/zane/session_1/ppt/.debs/root/usr/lib/x86_64-linux-gnu \
  npx slidev export
```

## 目录结构

- `slides.md` — 全部 26 页内容
- `style.css` — 全局字体覆盖（内嵌 Noto Sans SC，**不依赖系统中文字体**，任何机器都能正常显示中文）
- `assets/videos/` — 演示视频（B9.mp4、A2.mp4 已由 HEVC 转码为 H.264，浏览器可解码；原始 HEVC 文件仍在 `../video/`）
- `assets/papers/` — 8 篇论文首页截图（PyMuPDF 渲染自 `../../paper/<工号>/`）
- `assets/pics/` — A2 架构图、GitHub 合作图

## 页面结构（26 页）

1. 封面 → 2. 汇报框架 → 3. Part 01 分节
4. 总体框架（闭环四阶段）→ 5–12. 实验①⑦④⑥②③⑤⑧（各 1 页，含演示视频或关键指标）
13–18. 实验⑨ 重点（任务 / 方法链 / 四方法对比 / 真机误差地板 / 关节闭环 88% / 成果演示）
19. 两条闭环 → 20. 结论 → 21. 论文报告（8 篇首页截图）
22. Part 02 分节 → 23. A2 架构图 → 24. A2 演示视频 → 25. GitHub 协作 → 26. 谢谢聆听
