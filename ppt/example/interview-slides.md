---
theme: seriph
background: ./cover.jpg
title: "Zhou Haoyang - Personal Introduction"
class: text-center
transition: slide-left
mdc: true
download: true
---

<style scoped>
.slidev-layout {
  background-image: url('./cover.jpg') !important;
  background-size: cover !important;
  background-position: center !important;
  background-repeat: no-repeat !important;
  width: 100% !important;
  height: 100% !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  align-items: center !important;
  color: white !important;
  text-shadow: 0 2px 10px rgba(0,0,0,0.5) !important;
}
.slidev-layout h1 {
  font-size: 4rem !important;
  font-weight: 500 !important;
  margin-bottom: 0.5rem !important;
  color: white !important;
}
.slidev-layout h2 {
  font-size: 1.5rem !important;
  font-weight: 400 !important;
  margin-bottom: 2rem !important;
  color: rgba(255,255,255,0.9) !important;
}
.slidev-layout .info {
  font-size: 1rem !important;
  line-height: 1.8 !important;
  color: rgba(255,255,255,0.85) !important;
  margin-top: 1.5rem !important;
}
.slidev-layout .github {
  font-size: 0.9rem !important;
  color: rgba(255,255,255,0.6) !important;
  margin-top: 1rem !important;
}
</style>

# Personal Introduction

## Zhou Haoyang

<div class="info">
Jilin University, School of Software<br>
Junior · GPA 3.79/4 · Top 5% · CET-6 549
</div>

<div class="github">
GitHub: github.com/ZHYsfl
</div>

---
layout: center
class: text-center
transition: slide-left
title: "Overview"
---

<h1 style="color: #5a7a8a; font-weight: 400; font-size: 2.5rem; margin-bottom: 4rem;">Overview</h1>

<div style="display: flex; justify-content: center; gap: 5rem;">

<div style="text-align: center; max-width: 320px;">
<div style="font-size: 1.5rem; font-weight: 500; color: #1e40af; margin-bottom: 0.5rem;">What I Can Offer</div>
<div style="font-size: 0.95rem; color: #6b7280;">Systems · Research Foundations · Mindset</div>
</div>

</div>

---
layout: center
class: text-center
transition: slide-left
title: "What I Can Offer"
---

<h1 style="color: #5a7a8a; font-weight: 400; font-size: 2.5rem; margin-bottom: 4rem;">What I Can Offer</h1>

<div class="grid grid-cols-3 gap-6 px-6" style="margin-top: 2rem;">

<div class="p-5 bg-blue-50 rounded-lg text-left">

<h3 class="text-lg font-semibold mb-4 text-blue-800 text-center">Systems</h3>

<div class="mb-3">
  <div class="font-medium">AgentGenesis</div>
  <div class="text-sm text-gray-600">Distributed evaluation: Go backend; Python workers (HTTP + gRPC); Docker</div>
</div>

<div class="mb-3">
  <div class="font-medium">ToolCallingGo</div>
  <div class="text-sm text-gray-600">Parallel scheduling with cascade termination</div>
</div>

<div>
  <div class="font-medium">EducationAgent</div>
  <div class="text-sm text-gray-600">Real-time coordination (Go high-performance concurrency)</div>
</div>

</div>

<div class="p-5 bg-green-50 rounded-lg text-left">

<h3 class="text-lg font-semibold mb-4 text-green-800 text-center">Research Foundations</h3>

<div class="mb-3">
  <div class="font-medium">RL Theory</div>
  <div class="text-sm text-gray-600">CleanRL PR #535, familiar with DQN/PPO/GRPO</div>
</div>

<div class="mb-3">
  <div class="font-medium">ML Foundations</div>
  <div class="text-sm text-gray-600">ML course <span class="font-semibold text-green-800">96.4/100</span> · DL theory &amp; model deployment</div>
</div>

<div>
  <div class="font-medium">Data & Training</div>
  <div class="text-sm text-gray-600">Dataset construction, Single-node Multi-GPU SFT/RL</div>
</div>

</div>

<div class="p-5 bg-orange-50 rounded-lg text-left">

<h3 class="text-lg font-semibold mb-4 text-orange-800 text-center">Mindset & Soft Skills</h3>

<div class="mb-3">
  <div class="font-medium">Self-driven Learning</div>
  <div class="text-sm text-gray-600">Autonomous, proactive problem solver</div>
</div>

<div class="mb-3">
  <div class="font-medium">Teamwork & Communication</div>
  <div class="text-sm text-gray-600">Cross-university collaboration, Git training</div>
</div>

<div>
  <div class="font-medium">Hands-on</div>
  <div class="text-sm text-gray-600">Willing to do dirty work, build from scratch</div>
</div>

</div>

</div>

---
layout: center
class: text-center
transition: slide-left
title: "Systems Track"
---

<div style="font-size: 2.5rem; font-weight: 400; color: #5a7a8a;">Systems</div>

<div style="font-size: 1.2rem; color: #6b7280; margin-top: 0.5rem;">AgentGenesis · ToolCallingGo · EducationAgent</div>

---
transition: slide-up
title: "AgentGenesis: Technical Brief"
mdc: false
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.6rem;">AgentGenesis: Technical Brief</div>

<div style="font-size: 0.9rem; color: #6b7280; margin-bottom: 0.9rem;">Go backend; Python workers (HTTP to backend, gRPC to sandboxes). Motivation, adapters, three-level protocol, collaboration.</div>

<div class="grid grid-cols-2 gap-4 text-left text-sm max-w-6xl mx-auto">

<div class="p-3 bg-blue-50 rounded-lg">
<div class="font-semibold text-blue-800 mb-1">1) Motivation</div>
<div class="text-gray-700 leading-snug">Current benchmarks are fragmented: evaluation environments are unstable, standards are inconsistent, and protocols are not unified. AgentGenesis unifies single-agent and multi-agent evaluation under one reproducible runtime contract.</div>
</div>

<div class="p-3 bg-green-50 rounded-lg">
<div class="font-semibold text-green-800 mb-1">2) Adapter pattern advantages</div>
<div class="text-gray-700 leading-snug"><code>UserAdapter</code> decouples problem API from queues/gRPC. New tasks only update adapter, not core runtime.</div>
</div>

<div class="p-3 bg-orange-50 rounded-lg">
<div class="font-semibold text-orange-800 mb-1">3) Three-level protocol</div>
<div class="text-gray-700 leading-snug">L1 single-agent action/observation loop; L2 orchestrated multi-agent protocol in one user sandbox (fan-out observations, fan-in actions); L3 connection-primitive based multi-agent protocol across sandbox boundaries.</div>
</div>

<div class="p-3 bg-slate-100 border border-slate-200 rounded-lg">
<div class="font-semibold text-slate-700 text-[0.95rem] mb-1">4) Impact & adoption</div>
<div class="text-slate-600 text-[0.82rem] leading-snug">Shared GitHub workflow across universities, with code review, tests, and docs-first engineering.</div>
<div class="text-slate-600 text-[0.78rem] leading-snug mt-1">
  <strong>Usage:</strong> 22 tasks · concurrent multi-testpoint evals · encrypted API-key mgmt via real-time gateway
</div>
<div class="text-slate-600 text-[0.8rem] leading-snug mt-1">
  Platform: <a class="text-slate-700 no-underline" style="text-decoration:none;border-bottom:none;" href="http://82.157.250.20/" target="_blank">82.157.250.20</a>
  &nbsp;|&nbsp;
  GitHub: <a class="text-slate-700 no-underline" style="text-decoration:none;border-bottom:none;" href="https://github.com/ZHYsfl/AgentGenesis" target="_blank">github.com/ZHYsfl/AgentGenesis</a>
</div>
</div>

</div>

---
transition: none
title: "AgentGenesis: Architecture Evolution"
---

<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.25rem;">AgentGenesis: Architecture Evolution Roadmap</div>

<AgentArchitecture />

<div class="mt-1 max-w-5xl mx-auto text-left text-[0.68rem] leading-snug text-gray-500">
  <div><span class="font-semibold text-gray-600">Evolution:</span> prototype for one task -> adapter abstraction -> standardized protocol stack -> scalable multi-problem evaluation.</div>
  <div><span class="font-semibold text-gray-600">Stack:</span> Go backend; Python workers (HTTP + gRPC); Docker.</div>
</div>

---
transition: slide-up
title: "ToolCallingGo: Design"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.6rem;">ToolCallingGo: Design</div>

<div style="font-size: 0.9rem; color: #6b7280; margin-bottom: 0.85rem;">
Go SDK for LLM tool-calling (openai-go/v3) · <a class="text-slate-700 no-underline" style="text-decoration:none;border-bottom:none;" href="https://github.com/ZHYsfl/tool-calling-go" target="_blank">github.com/ZHYsfl/tool-calling-go</a>
</div>

<div style="font-size: 0.88rem; color: #374151; margin-top: -0.2rem; margin-bottom: 0.75rem;"><strong>Motivation:</strong> Multi-agent systems need strong concurrency primitives, but the Go agent/tool-calling ecosystem is still weak, so we built this shovel.</div>

<div class="rounded-xl border border-slate-300 bg-white px-4 py-3 text-left text-[0.86rem] leading-snug mb-3">
<span class="font-semibold text-slate-700">Runtime flow:</span>
<code>Agent.Chat(...)</code> (built-in tool loop + parallel tool calls) -> <code>Batch</code> (session-level fan-out) -> <code>BatchRace</code> (race-to-success + cascading cancellation)
</div>

<div class="grid grid-cols-2 gap-3 text-left">

<div class="rounded-lg bg-blue-50 border border-blue-100 p-3">
<div class="text-blue-800 font-semibold mb-1">Core SDK</div>
<div class="text-[0.84rem] text-slate-700"><code>Agent</code> + <code>Tool</code> + <code>ToolFunc(ctx, args)</code></div>
<div class="text-[0.8rem] text-slate-600 mt-1">Clean interface for tool registration and execution loop.</div>
</div>

<div class="rounded-lg bg-green-50 border border-green-100 p-3">
<div class="text-green-800 font-semibold mb-1">Parallel primitives</div>
<div class="text-[0.84rem] text-slate-700"><code>getToolResponseObservations</code> · <code>Batch(maxConcurrent)</code> · <code>BatchRace(SuccessCondition)</code></div>
<div class="text-[0.8rem] text-slate-600 mt-1">Tool-level parallelism inside one chat + session-level parallelism across chats.</div>
</div>

<div class="rounded-lg bg-orange-50 border border-orange-100 p-3">
<div class="text-orange-800 font-semibold mb-1">Reliability</div>
<div class="text-[0.84rem] text-slate-700"><code>WithMaxToolRetries</code> · <code>WithDebug</code></div>
<div class="text-[0.8rem] text-slate-600 mt-1">Retry strategy and debuggability built into agent options.</div>
</div>

<div class="rounded-lg bg-slate-100 border border-slate-200 p-3">
<div class="text-slate-800 font-semibold mb-1">Observability</div>
<div class="text-[0.84rem] text-slate-700"><code>started/success/no_match/error/cancelled</code></div>
<div class="text-[0.8rem] text-slate-600 mt-1">Event stream for orchestration progress and failure diagnosis.</div>
</div>

</div>

---
transition: slide-up
title: "EducationAgent: Voice Interrupt Pipeline"
---

<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.45rem;">EducationAgent: Voice Interrupt Pipeline</div>

<div class="w-full flex justify-center">
  <div class="text-center">
    <div class="text-xs text-gray-600 mb-1">Protocol View (interrupt + context/tool signals)</div>
    <img src="/voiceagent架构图.png" alt="EducationAgent voice protocol graph" style="max-height: 475px; width: auto; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px;" />
  </div>
</div>
---
transition: slide-up
title: "EducationAgent: Runtime Graph"
---

<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.45rem;">EducationAgent: Runtime Graph</div>

<div class="w-full flex justify-center">
  <img src="./EducationAgent架构图.png" alt="EducationAgent runtime graph" style="max-height: 455px; width: auto; border: 1px solid #e5e7eb; border-radius: 10px;" />
</div>

---
transition: slide-up
title: "EducationAgent: PPT Agent Three-Way Merge"
---

<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.45rem;">EducationAgent: PPT Agent Three-Way Merge</div>

<div class="w-full flex justify-center">
  <img src="./ThreeWayMergeAlgorithm.png" alt="PPT Agent three-way merge algorithm" style="max-height: 445px; width: auto; border: 1px solid #e5e7eb; border-radius: 10px;" />
</div>

---
layout: center
class: text-center
transition: slide-left
title: "Research Foundations Track"
---

<div style="font-size: 2.5rem; font-weight: 400; color: #5a7a8a;">Research Foundations</div>

<div style="font-size: 1.2rem; color: #6b7280; margin-top: 0.5rem;">RL Theory · ML Foundations · Data & Training</div>

---
layout: center
class: text-center
transition: slide-up
title: "RL: reading & contribution"
---

<div style="font-size: 2.5rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.35rem;">RL</div>

<div style="font-size: 0.95rem; color: #6b7280; margin-bottom: 1rem;">DQN / PPO / GRPO — code & theory; one CleanRL PR on buffers (row below).</div>

<div class="max-w-3xl mx-auto text-left text-sm">

| Topic | Notes |
|------|------|
| **DQN / PPO** | Code + Theory |
| **[PR #535](https://github.com/vwxyzjn/cleanrl/pull/535)** | <code>cleanrl_utils/buffers.py</code> · <code>swap_and_flatten</code> — explicit reshape, validated inputs, stable outputs. (Python philosophy: explicit is better than implicit — for tensor shapes here.) |
| **GRPO** | Theory + Unsloth |

<p class="mt-4 text-gray-800 m-0"><strong>Applied:</strong> SFT + RL on BRIGHT reranking.</p>

</div>

---
layout: center
class: text-center
transition: slide-up
title: "RL: notes sample"
---

<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.35rem;">RL study notes</div>

<div style="font-size: 0.9rem; color: #6b7280; margin-bottom: 0.5rem;">~44 pages handwritten (theory, derivations, algorithms). Six sample pages.</div>

<div class="text-[0.72rem] text-gray-500 mb-4">Full RL scans live in <a class="text-slate-700 no-underline" style="text-decoration:none;border-bottom:none;" href="https://github.com/ZHYsfl/Notebooks/tree/main/RL%E7%AC%94%E8%AE%B0" target="_blank"><code>RL笔记</code></a> · <a class="text-slate-700 no-underline" style="text-decoration:none;border-bottom:none;" href="https://github.com/ZHYsfl/Notebooks" target="_blank">github.com/ZHYsfl/Notebooks</a></div>

<div class="grid grid-cols-3 gap-x-4 gap-y-3 w-full items-start max-w-6xl mx-auto px-2">
  <div class="text-center">
    <div class="text-xs text-gray-500 mb-1.5 leading-tight">Temporal difference (TD)</div>
    <img src="./时序差分算法.jpg" alt="Handwritten TD algorithm notes" style="max-height: 188px; width: 100%; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px;" />
  </div>
  <div class="text-center">
    <div class="text-xs text-gray-500 mb-1.5 leading-tight">REINFORCE</div>
    <img src="./REINFORCE.jpg" alt="Handwritten REINFORCE notes" style="max-height: 188px; width: 100%; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px;" />
  </div>
  <div class="text-center">
    <div class="text-xs text-gray-500 mb-1.5 leading-tight">Actor–Critic</div>
    <img src="./ActorCritic笔记.jpg" alt="Handwritten Actor-Critic RL notes" style="max-height: 188px; width: 100%; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px;" />
  </div>
  <div class="text-center">
    <div class="text-xs text-gray-500 mb-1.5 leading-tight">DQN</div>
    <img src="./DQN笔记.jpg" alt="Handwritten DQN notes" style="max-height: 188px; width: 100%; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px;" />
  </div>
  <div class="text-center">
    <div class="text-xs text-gray-500 mb-1.5 leading-tight">PPO (penalty)</div>
    <img src="./ppo-penalty.jpg" alt="Handwritten PPO-penalty notes" style="max-height: 188px; width: 100%; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px;" />
  </div>
  <div class="text-center">
    <div class="text-xs text-gray-500 mb-1.5 leading-tight">PPO (clipping)</div>
    <img src="./ppo-clipping.jpg" alt="Handwritten PPO-clipping notes" style="max-height: 188px; width: 100%; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px;" />
  </div>
</div>

---
layout: center
class: text-center
transition: slide-up
title: "ML: overview"
---

<div class="mx-auto box-border w-full" style="max-width: 820px; padding-left: 56px; padding-right: 56px;">

<div style="font-size: 1.95rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.35rem;">Machine Learning</div>

<div class="mb-3 text-sm font-semibold text-green-800">ML course · <span style="font-size: 1.1rem;">96.4/100</span></div>

<div style="font-size: 0.8rem; color: #6b7280; margin-bottom: 0.85rem; line-height: 1.45;">First principles &amp; systematic notes — outline on the left; full map on the right.</div>

<div class="flex w-full flex-row items-start" style="gap: 22px;">
  <div class="shrink-0 text-[0.72rem] text-gray-700" style="width: 220px;">
    <ul class="m-0 list-disc space-y-1 pl-5 text-left leading-snug marker:text-gray-400">
      <li>Linear Regression</li>
      <li>Logistic Regression</li>
      <li>Perceptron</li>
      <li>Backpropagation (BP)</li>
      <li>RBF Networks</li>
      <li>K-Means</li>
      <li>Self-Organizing Competitive Nets (SOM)</li>
      <li>Affinity Propagation (AP)</li>
      <li>DBSCAN</li>
      <li>Hierarchical Clustering</li>
      <li>SVM</li>
      <li>EM</li>
      <li>PCA</li>
    </ul>
  </div>
  <div class="min-w-0 flex-1">
    <div class="flex min-h-0 w-full items-center justify-center overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm" style="height: 268px; max-height: 268px;">
      <img src="./machine_learning.svg" alt="Machine learning knowledge map" style="max-width: 100%; max-height: 100%; width: auto; height: auto; object-fit: contain; display: block;" />
    </div>
  </div>
</div>

<div class="text-[0.72rem] text-gray-500 mt-4 text-center leading-relaxed">Same file in repo: <a class="text-slate-700 no-underline" style="text-decoration:none;border-bottom:none;" href="https://github.com/ZHYsfl/Notebooks/blob/main/machine_learning.svg" target="_blank"><code>machine_learning.svg</code></a> · <a class="text-slate-700 no-underline" style="text-decoration:none;border-bottom:none;" href="https://github.com/ZHYsfl/Notebooks" target="_blank">github.com/ZHYsfl/Notebooks</a></div>

</div>

---
layout: center
class: text-center
transition: slide-up
title: "ML: notes"
---

<div class="mx-auto w-full max-w-4xl px-6 box-border">

<div style="font-size: 1.85rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.5rem; line-height: 1.25;">ML study notes</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 0.6rem; line-height: 1.45;">~70 pages handwritten (theory, derivations, algorithms). Electronic notes — neural networks and clustering.</div>

<div class="text-[0.78rem] text-gray-500 mb-5 leading-relaxed">Full ML scans in <a class="text-slate-700 no-underline" style="text-decoration:none;border-bottom:none;" href="https://github.com/ZHYsfl/Notebooks/tree/main/ML%E7%AC%94%E8%AE%B0" target="_blank"><code>ML笔记</code></a> · <a class="text-slate-700 no-underline" style="text-decoration:none;border-bottom:none;" href="https://github.com/ZHYsfl/Notebooks" target="_blank">github.com/ZHYsfl/Notebooks</a></div>

<div class="grid grid-cols-2 gap-x-6 gap-y-4 w-full items-start">
  <div class="text-center flex flex-col">
    <div class="text-xs text-gray-500 mb-2 font-medium">Neural networks</div>
    <img src="./神经网络（部分）.png" alt="Neural network study notes" class="w-full object-contain rounded-xl border border-gray-200 shadow-sm" style="max-height: 240px; height: auto;" />
  </div>
  <div class="text-center flex flex-col">
    <div class="text-xs text-gray-500 mb-2 font-medium">Clustering</div>
    <img src="./聚类.png" alt="Clustering study notes" class="w-full object-contain rounded-xl border border-gray-200 shadow-sm" style="max-height: 240px; height: auto;" />
  </div>
</div>

</div>

---
layout: center
class: text-center
transition: slide-up
title: "ML: notes · handwritten"
---

<div class="mx-auto w-full max-w-4xl px-6 box-border">

<div style="font-size: 1.85rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.45rem; line-height: 1.25;">ML study notes</div>

<div style="font-size: 0.88rem; color: #6b7280; margin-bottom: 0.75rem; line-height: 1.45;">Handwritten pages — SVM and PCA (same folder as above).</div>

<div class="grid grid-cols-2 gap-x-6 gap-y-4 w-full items-start">
  <div class="text-center flex flex-col">
    <div class="text-xs text-gray-500 mb-2 font-medium">SVM</div>
    <img src="./SVM.jpg" alt="Handwritten SVM notes" class="w-full object-contain rounded-xl border border-gray-200 shadow-sm" style="max-height: 260px; height: auto;" />
  </div>
  <div class="text-center flex flex-col">
    <div class="text-xs text-gray-500 mb-2 font-medium">PCA</div>
    <img src="./PCA.jpg" alt="Handwritten PCA notes" class="w-full object-contain rounded-xl border border-gray-200 shadow-sm" style="max-height: 260px; height: auto;" />
  </div>
</div>

</div>

---
layout: center
class: text-center
transition: slide-left
title: "Mindset Track"
---

<div style="font-size: 2.5rem; font-weight: 400; color: #5a7a8a;">Mindset</div>

<div style="font-size: 1.2rem; color: #6b7280; margin-top: 0.5rem;">Self-driven · Teamwork · Hands-on</div>

---
transition: slide-up
title: "Mindset - Evidence"
---

<div style="font-size: 2.2rem; font-weight: 400; color: #5a7a8a; margin-bottom: 0.5rem;">Mindset: Evidence</div>

<div style="font-size: 0.95rem; color: #6b7280; margin-bottom: 0.8rem;">Consistent habits that compound over time</div>

<div class="grid grid-cols-3 gap-4 text-left">

<!-- Column 1: Library + Communication -->
<div class="flex flex-col gap-3">
<div class="text-center">
<div class="rounded-lg border border-slate-200 overflow-hidden mb-2" style="height: 140px;">
<img src="/图书馆打卡记录.jpg" alt="Library check-ins" style="width: 100%; height: 100%; object-fit: cover; object-position: 50% 14%;" />
</div>
<div class="font-semibold text-slate-700" style="font-size: 0.9rem;">800+ library check-ins</div>
<div class="text-slate-500" style="font-size: 0.75rem;">Screenshot: 741 (Nov 2025) · Now: 800+</div>
</div>
<div class="p-3 bg-blue-50 rounded border border-blue-100">
<div class="font-semibold text-blue-800 mb-1" style="font-size: 0.85rem;">Study Notes</div>
<div style="font-size: 0.8rem; line-height: 1.4; color: #1e40af;">RL, ML, Data Structure, Agent, etc.</div>
<div style="font-size: 0.75rem; margin-top: 0.2rem;"><a href="https://github.com/ZHYsfl/Notebooks" target="_blank" class="text-blue-600 hover:underline">github.com/ZHYsfl/Notebooks</a></div>
</div>
</div>

<!-- Column 2: Dev Logs 1 -->
<div class="text-center">
<div class="rounded-lg border border-slate-200 overflow-hidden mb-2" style="height: 240px;">
<img src="/一些开发日志1.png" alt="Dev logs 1" style="width: 100%; height: 100%; object-fit: cover; object-position: top;" />
</div>
<div class="font-semibold text-slate-700" style="font-size: 0.9rem;">Daily dev logs (1)</div>
<div class="text-slate-500" style="font-size: 0.8rem;">Systematic reflection on debugging & design</div>
</div>

<!-- Column 3: Dev Logs 2 + Communication + Notes -->
<div class="flex flex-col gap-2">
<!-- Dev Logs 2 -->
<div class="text-center">
<div class="rounded-lg border border-slate-200 overflow-hidden mb-1.5" style="height: 110px;">
<img src="/一些开发日志2.png" alt="Dev logs 2" style="width: 100%; height: 100%; object-fit: cover; object-position: top;" />
</div>
<div class="font-semibold text-slate-700" style="font-size: 0.85rem;">Daily dev logs (2)</div>
<div class="text-slate-500" style="font-size: 0.75rem;">Documentation & iteration</div>
</div>
<!-- Communication -->
<div class="text-center">
<div class="rounded-lg border border-slate-200 overflow-hidden mb-1.5" style="height: 75px;">
<img src="/交流思维.png" alt="Communication mindset" style="width: 100%; height: 100%; object-fit: cover; object-position: 50% 20%;" />
</div>
<div class="font-semibold text-slate-700" style="font-size: 0.85rem;">Active communication</div>
<div class="text-slate-500" style="font-size: 0.75rem;">Weekly deep conversations</div>
</div>
<!-- Study Notes -->
<div class="p-2.5 bg-green-50 rounded border border-green-100">
<div class="font-semibold text-green-800 mb-0.5" style="font-size: 0.8rem;">Git Workshops</div>
<div style="font-size: 0.75rem; line-height: 1.35; color: #166534;">Led hands-on Git training → adopted GitHub Flow</div>
<div style="font-size: 0.7rem; margin-top: 0.15rem;"><a href="https://github.com/ZHYsfl/cowork" target="_blank" class="text-green-600 hover:underline">cowork</a> · <a href="https://github.com/ZHYsfl/Cowork_" target="_blank" class="text-green-600 hover:underline">Cowork_</a></div>
</div>
</div>

</div>
---
layout: center
class: text-center
transition: slide-left
title: "Future Plan"
---

<h1 style="color: #5a7a8a; font-weight: 400; font-size: 2.5rem; margin-bottom: 2rem;">Future Plan</h1>

<div class="max-w-3xl mx-auto px-6 text-left" style="font-size: 0.88rem; line-height: 1.45;">
<div class="p-6 bg-green-50 rounded-lg border border-green-100">

<div class="mb-4">
  <div class="font-medium text-green-900">Why PhD</div>
  <div class="text-sm text-gray-600 mt-1">Both academia and industry require the ability to tackle complex problems. The prerequisite is doing solid research during these years — building the foundation to solve real challenges, wherever that leads.</div>
</div>
<div class="mb-4">

  <div class="font-medium text-green-900">From now</div>
  <div class="text-sm text-gray-600 mt-1">Ready to contribute <strong>immediately</strong>. If I get the opportunity, I can <strong>join the lab in senior year (Year 4)</strong> and start working with the group right away.</div>
</div>

<div class="mb-4">
  <div class="font-medium text-green-900">Year 1</div>
  <div class="text-sm text-gray-600 mt-1">Get deep into the <strong>lab codebase and projects</strong>; shore up <strong>OS and networking</strong> fundamentals; <strong>reproduce classic papers</strong>; <strong>help with lab work</strong>; <strong>narrow down a research direction</strong> — through <strong>independent exploration</strong> or <strong>with guidance from the lab / faculty</strong>.</div>
</div>

<div class="mb-3">
  <div class="font-medium text-green-900">Current research interests</div>
  <div class="text-sm text-gray-600 mt-2 space-y-2">
    <div><span class="text-green-800 font-medium">1.</span> <strong>Agent infra</strong> / <strong>Agent systems</strong> that run <strong>reliably</strong>.</div>
    <div><span class="text-green-800 font-medium">2.</span> <strong>RL infra</strong> — accelerating <strong>large-model training &amp; inference</strong> stacks.</div>
  </div>
  <div class="text-xs text-gray-500 mt-3" style="line-height: 1.4;">These are <strong>starting points</strong> — I'm <strong>open</strong> to other directions that fit the lab; not ruling anything out upfront.</div>
</div>

</div>
</div>

---
transition: fade
class: text-center
title: "Thank You"
---

# Thank You

<div class="mt-8 text-lg">

My goal: AI Systems that don’t just run — they run **reliably**.<br>

</div>

<div class="mt-6 text-base opacity-80">

“Environment alone is an uncontrollable genius; Harness alone is a safe mediocrity.”

</div>

<div class="mt-8">

**Zhou Haoyang**

School of Software, Jilin University · Junior

📧 <a href="mailto:zhouhy5523@mails.jlu.edu.cn" class="text-slate-700 no-underline hover:opacity-90" style="text-decoration: none; border-bottom: none;">zhouhy5523@mails.jlu.edu.cn</a><br>
📱 13223291973<br>
🔗 <a href="https://github.com/ZHYsfl" target="_blank" rel="noopener noreferrer" class="text-slate-700 no-underline hover:opacity-90" style="text-decoration: none; border-bottom: none;">github.com/ZHYsfl</a>

</div>
