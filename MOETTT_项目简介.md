# MOETTT 项目简介

## 1. 项目定位

`MOETTT`（当前代码主体在 `hotpot_param_mem/`）是一个面向**多跳问答（multi-hop QA）**的实验型框架，核心目标是：

- 让模型按照“**搜索 -> 累积证据 -> 决策结束 -> 输出答案**”的方式逐步推理；
- 在推理过程中支持“**在线记忆更新**”（LoRA/QLoRA 方向），把当前轮次抽取出的关键信息注入到可训练适配器；
- 支持本地推理与集中式推理服务两种模式，并支持多进程/多卡协同运行。

从工程角度看，它是一个“**检索增强 + 可选在线参数记忆**”的问答执行系统。

---

## 2. 主要执行流程（端到端）

1. 入口脚本 `scripts/run.py` 解析参数，构建 `RunConfig`。
2. 读取数据集（如 Hotpot 风格样本），生成带 `episode_id` 的样本列表。
3. `multiproc.py` 按 GPU / worker 规划任务：
   - 可选启动集中式推理服务；
   - 启动多个 worker 并行执行样本。
4. 每个 worker 在 `runner.py` 中循环处理样本：
   - 第一步强制发起搜索；
   - 后续由模型输出 `<search>...</search>` 或 `<finish/>`；
   - 调用本地检索，更新 history；
   - 可选触发记忆训练并保存本地 adapter 版本；
   - 达到同步条件后聚合各 worker adapter 状态，发布全局轮次。
5. 最终写出：
   - step 级 trace；
   - 题目级评测结果（EM/F1）；
   - 汇总 summary。

---

## 3. 代码结构与模块职责

## `scripts/run.py`
- 命令行入口，负责参数定义、随机种子设置、配置构建、启动多进程执行。

## `hotpot_param_mem/config.py`
- 定义 `RunConfig` 配置数据类；
- 管理输出目录、adapter 发布/暂存目录、推理模式与训练模式等开关。

## `hotpot_param_mem/data.py`
- 读取样本，生成稳定 `episode_id`；
- 提供记录去重与排序辅助函数。

## `hotpot_param_mem/multiproc.py`
- 多进程调度核心：
  - 解析 GPU 分配；
  - 构建 worker plan；
  - 可选拉起推理 HTTP 服务；
  - 回收进程并合并输出结果。

## `hotpot_param_mem/runner.py`
- 单 worker 执行核心逻辑：
  - 逐步动作生成（search/finish）；
  - 本地检索证据注入 history；
  - 触发 memory update 与 adapter 版本推进；
  - 记录 trace 与 eval。

## `hotpot_param_mem/services.py`
- 统一封装服务层：
  - `GenerationService`：推理生成动作与最终答案；
  - `MemoryUpdateService`：在线记忆更新；
  - `AdapterRuntime`：管理本地版本、全局同步轮次、聚合发布。

## `hotpot_param_mem/llm_vllm.py`
- vLLM 相关能力：
  - 本地推理引擎封装；
  - LoRA adapter 动态切换；
  - 推理 HTTP 服务（`/health`, `/generate`）及子进程启动与就绪等待。

## `hotpot_param_mem/mem_injector_ntp.py`
- 记忆注入训练器：
  - 从证据中抽取 snippet；
  - 触发 LoRA/QLoRA 更新；
  - 保存 adapter 与最终合并导出。

## `hotpot_param_mem/env_local.py`
- 本地检索逻辑：根据 query 在样本 context 中召回段落，并生成信息块。

## `hotpot_param_mem/prompts.py`
- 管理动作层/答案层/压缩层提示模板：
  - 行为协议标签（`<search>`, `<finish/>`, `<answer>`）；
  - 限制输出格式，降低解析失败率。

## `hotpot_param_mem/parsing.py`
- 对模型输出进行结构化解析：
  - action 解析（search/finish）；
  - final answer 解析；
  - 解析失败时给出强制终止标记。

## `hotpot_param_mem/metrics.py`
- 实现 EM/F1 指标计算。

## `hotpot_param_mem/logger.py`
- JSONL 读写与 summary 汇总。

---

## 4. 关键设计特点

- **协议化推理动作**：通过标签约束减少自由文本噪声，便于稳健解析。
- **可选在线记忆**：在推理中增量更新 adapter，使后续步骤可利用新记忆。
- **本地/集中式推理兼容**：既支持单机本地推理，也支持集中服务 + worker 客户端模式。
- **多进程可扩展**：按 worker 并发处理样本，支持阶段性同步与最终聚合。
- **可追踪可评估**：完整产出 trace/eval/summary，方便实验分析与复现。

---

## 5. 适用场景（建议）

- 多跳问答与检索增强推理实验；
- 在线学习/持续学习（adapter 级）原型验证；
- 比较“无记忆更新”与“在线记忆更新”在 EM/F1 与步数上的差异；
- 研究分布式推理与训练资源解耦（inference/training 分卡）的工程方案。

---

## 6. 一句话总结

`MOETTT` 是一个把“**结构化搜索决策**、**本地证据检索**、**在线 LoRA 记忆更新**、**多进程协同执行**”整合在一起的多跳问答实验框架。
