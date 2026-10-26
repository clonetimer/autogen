# PHM AutoDesign 0.8.0 Stable Core

**Current release scope:** reproducibly installable and regression-tested `phm-design` software framework, including the previously limited `0.7.0` SAT/Basilisk offline pathways, R93 source-data contracts, and R95 **SOURCE-only design proposals**. R94 native `torch_like` training/persistence depends on a **private optional package** and is **NOT part of the 0.8.0 Stable Core acceptance**; AAS application integration and independently qualified industrial model performance remain experimental. No actual equipment/flight-control use is qualified.

See [0.8.0 official scope and verification](docs/STABLE_080_RELEASE.md). Older release and development narratives below are **historical documentation**, not current package status or test counts.

---

## Historical research / developer notes

# 0.7.0.dev10 — R51 研发候选（非 RC / Stable）

本轮新增按外部工况族留出验证、显式数值秩支持描述、声明观测合同下的精确标签冲突审计。默认世界分组行为保持兼容，Stable0.6.0与已冻结dev9未覆盖。

```python
from phm_design.open_design_fit import design
# examples/spec/classes 由调用者按部署观测合同构造；family_map 来自离线工况元数据。
model, report = design(examples, spec, classes, healthy_label=0,
                       validation_families=family_map, max_trials=16)
```

family_map 必须完整覆盖训练数据的世界分组；工况族不传入预测。design 的 max_trials 不是硬 CPU 预算；研究硬预算由上层 supervisor 执行。未找到合格模型不等于证明任务不可行；无精确观测冲突也不等于证明任务可行。

以下保留原产品说明，历史版本称谓仅作上下文，资格以本轮交接状态为准。

---
# PHM AutoDesign 0.7.0.dev6（研发版）

**唯一正式 Stable 仍为 `phm-design 0.6.0`。** `0.7.0.dev6` 是开发分支，**不是 RC 或 Stable**。

本版新增可安装的 `phm-design-readiness`：接收经过 SHA256 绑定的 development-only 评估证据，检查各环境的故障 Recall、正常误报、有效覆盖率、样本支持数和缺失的 world-group 信息，输出有依据的下一批实验需求；**不会用故障真值做部署推理，不会自动授予诊断模型资格**。

R17 真实证据回放覆盖 Thermal R16 及 ADCS R3 两项任务，均返回 `HOLD_DO_NOT_PROMOTE`。配对物理信息审计仅使用已公开的18个 Thermal 训练世界，不构成新 blind。新增研发测试19项通过，定向兼容回归58项通过；**506项完整测试尚未跑完，不得声称全量通过**。详情参见 `DESIGN_READINESS_DEV6.md` 及独立 R17 研究证据包。

---

## 历史版本记录：0.7.0.dev5

# PHM AutoDesign 0.7.0.dev5（研发版）

**唯一正式稳定版：`0.6.0 Stable`。本版 `0.7.0.dev5` 不是 RC/Stable。**

本版新增一个可审计的 Temporal Algorithm-DAG 子图：签名物理残差 → EWMA / 双侧 CUSUM / 滑动统计 → 观测有效性检查 → 连续时间报警，并提供节点源码、策略指纹和专项回归测试。

**验证边界：** R7/R8 基于文献方法的 Thermal 小样本渐进退化回放均为 FAIL-CLOSED；新增设计能力**不代表**压力任务已合格。详细设计、参考文献、API 示例见 `docs/TEMPORAL_ALGODAG_DEV5.md`。历史版本说明在下面，不是此版新资格声明。

---

## 历史说明：0.7.0.dev1

### 当时的 0.7.0.dev1 开发状态（历史）

`phm-design 0.6.0 Stable` remains the frozen Stable release. `0.7.0.dev1` is the active development line for high-level Algorithm-DAG synthesis.

The current ADCS V15 development candidate passed a fresh pre-frozen blind (96 cases / 48 groups) with Macro-F1 **0.98436**, fault-detection recall **1.0**, wheel-localization accuracy **1.0**, and a strict gate PASS. The complete Basilisk-enabled regression is **462/462 PASS**. This is development evidence, not a `0.7.0` Stable release or hardware/flight qualification.

See `ALGORITHM_DAG_DEV41.md` and `ALGORITHM_DAG_070DEV1_QUALIFICATION_20261008.json`.

The formal workflow is exposed through `phm-design-algorithm-dag` with `single`, `multi`, `freeze`, and `evaluate` modes.

---

# PHM Design 0.6.0.dev7

EXPERIMENTAL / RESEARCH_ONLY。

> **当前交付：sat-platform 0.7.8 源码 + PHM 0.6.0.dev7 研究扩展。** 本轮从已校验的 dev3 制品恢复可见增量；dev5 临时源码与权重未在当前环境恢复。下面的旧版批准和发布说明均为历史语境，不代表本包全量门禁或真实设备资格。使用入口见 `RECOVERY_AND_BENCHMARK_DEV6.md`。


## 当前入口

`python -m phm_design --help` 保留组合设计、鲁棒输入、机理与参考修正入口。新 `select-reference-route` / `evaluate-selected-reference` / `predict-selected-reference` 使用已冻结的参考、固定或生成路线。

先阅读 `RECOVERY_AND_BENCHMARK_DEV6.md`；完整分因素运行使用 `scripts/run_dev6_study.py`。

核心源码包不含运行wheel、原始数据或权重；模型和数据在独立证据包。新增源恢复版本不会自动继承旧报告中的成绩。

## 安装与测试

```bash
python -m pip install --no-build-isolation --no-deps .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/phm_design
```

这些命令假设核心依赖已装好；`--no-deps` 不意味着依赖约束已满足。Basilisk是可选仿真依赖。无Basilisk时原生测试明确跳过。

历史指南保留供旧模型复算，不覆盖本版本的行为说明。


## dev7：独立选择性风险校准

`calibrate-risk` 在结构与路线冻结后，使用独立 calibration cohort 选择拒答阈值；
`evaluate-risk` 只在阈值制品密封后打开 test。风险统计以 world group 为单位，
使用单侧精确二项 Clopper–Pearson 上界，并对预声明阈值网格做 Bonferroni 修正。
若没有阈值同时满足风险上界与最低覆盖率，系统 fail-closed 为全部拒答。

路线选择用的 calibration 数据不得复用作风险校准。该机制不是 softmax 概率校准、
真实硬件风险认证或生产资格。详见 `SELECTIVE_RISK_DEV7.md`。

执行环境不稳定时，先阅读 `CAAS_EXECUTION_WORKAROUND_DEV7.md`。


## 0.6.0.dev10 hybrid commissioning arm

Preserves the dev7 mechanism representation and optionally appends compact residual/context features fitted from an independent healthy commissioning run. The arm is enabled only when development-validation and route-selection calibration satisfy per-class noninferiority plus strict worst-environment improvement. The formal dev10 validation rejected the hybrid arm in all three domains; it remains an experimental candidate, not a default.
