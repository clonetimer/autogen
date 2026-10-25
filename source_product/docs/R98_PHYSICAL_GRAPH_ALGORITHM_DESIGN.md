# R98 — 物理条件门控的诊断算法结构设计

**2026-10-25 / Draft Research / 建基于 R97 → R96 → R95 → R94 → R93 研究链。** Stable `main` 的限定卫星仿真离线资格不变。

## 1. 设计目标和工程边界

`autogen` 的核心职责是根据设备能力、任务、可用测量及试验成本，**组织和选择诊断算法结构**，而不是再实现已有分类器、异常阈值或平台数据管理。R98 将 R97 的 FFT 特征扩展为只有满足物理前提才可用的候选图：

```text
R93 设备/数据/任务约束（只接受 SOURCE TRAIN）
    │
    ├─ A: 时域 RMS / 峰值系数                   [始终保留可比基线]
    ├─ B: 共振带通 → Hilbert 包络 → 包络调制特征 [采样率满足要求]
    ├─ C: 独立转速输入 → FFT 转速阶次能量         [必须具备每窗口转速观测]
    └─ D: B + C + 同一记录内标量特征拼接         [必须额外有已登记标量]
    │
    ▼
R96 每一个 SOURCE 留组 TRAIN 折单独拟合预处理
    ▼
R94 torch_like 已有实际模型提供者（不会再开发诊断器）
    ▼
R95 相同组划分的 SOURCE TRAIN 内部探索比较
    ▼
临时候选短名单 / 证据不足拒绝，不得称独立验证或生产合格
```

只有**真实声明的转速传感器**在 EquipmentCard 注册为 `modality="speed"`、`unit="rpm"`，且被 TrialSpec 明确允许使用时，才生成阶次图 C/D。**绝不从设备名称、振动信号的主峰或分类标签猜测速。** 没有转速输入时合法图只包括 A/B；没有独立标量观测时，即便有转速也不生成 D。

## 2. 物理约束（固定、可审计，不开放任意代码插件）

- 原始振动信号遵循 R97：**单一**合法振动通道，标称 20–200kHz 采样，原始 `sample_rate_hz` 与设备声明严格一致，单条原始记录 4096 至 1,000,000 个有限样本；技术窗口固定 4096 个点且不重叠，聚合时用中位数，每个 `record_id` **只生成一条特征记录**。
- 包络图采用 SciPy 的**固定四阶 2000–8000Hz Butterworth 带通**、每个非重叠窗口 `sosfiltfilt`、Hilbert 解析信号求包络，去均值后对包络进行 Hann 加窗 FFT。新增指标为 **40–700Hz 调制能量占 40–2000Hz 能量比例**，以及 40–700Hz 峰频中位数。
- 阶次图在每个 4096 点窗口使用 `rpm_per_window` 和**独立测速测量**声明 `rpm_provenance="tachometer_per_window_self_declared"`。RPM 必须 120–12000、每窗口数量准确、窗间相对中位数变动不超过 **3%**。如果变化超过 3%，因为未实现变速重采样阶次跟踪，明确拒绝，而不是强行解释为可靠阶次谱。
- 阶次谱在每窗口先去均值、Hann FFT，再用 **frequency_hz ÷ (rpm/60)** 构造阶次坐标，计算 1–3 阶与 3–7 阶能量占 0.5–10 阶能量的比例。每窗口每个阶次区间至少要有两个 FFT 频率点才允许评价，否则拒绝。注意：这些是**宽阶次能量描述**，**不是**已知轴承几何的 BPFI/BPFO 或具体故障理论频率。
- D 的“传感器融合”**只在原始记录层**连接 A+B+C 提取的一行特征与其他已登记的标量汇总通道。要求源记录显式声明 `scalar_alignment="same_recording_interval_self_declared"`；这不是真实校验的逐采样同步，也未处理时钟漂移、插值和单位校准，不得称为严格同步融合。
- 验证值、事件标签、真实故障位置、物理设备 ID、未来故障时间、未登记传感器、非法测速来源或任意 Python 表达式不可进入特征。所有原始输入在**首次算法 fit 之前**统一检查，任何非法原始记录都会中止整份实验。
- 固定定义的结构图与 R95/R96 算法/传感器/预处理选择以 SHA-256 锁定；不同图**分享一个** `TrialSpec.max_trials`，每种图先给所有可执行家族同等基础候选名额，不允许把一个新增模块默默扩展为额外免费训练次数。最多总共 64 个 SOURCE 留组训练 fit。
- 不实现复杂增量神经网络、不开放新网络数学原型、不代替 AAS 对异常的连续点、阈值、报警延迟等决策。算法后处理仍由实际算法提供者拥有。

## 3. 运行输入合同：两条差异很重要

针对已登记 tachometer 的任务，R98 的一条原始记录结构为：

```json
{
  "record_id": "r001",
  "sample_rate_hz": 48000,
  "samples": [0.0, 0.1],
  "scalars": {"motor_current": 1.0},
  "scalar_alignment": "same_recording_interval_self_declared",
  "rpm_per_window": [3000.0, 3000.0],
  "rpm_provenance": "tachometer_per_window_self_declared"
}
```

上例 `samples` 仅用于**说明字段**；真实执行最少需要 4096 个完整样本，示例两次 RPM 代表恰好两个 4096 点窗口（至少 8192 点），否则拒绝。数值数组 SHA 属于数字样本，而**不等于原始 MAT/CSV 二进制 SHA**。

没有转速通道的任务严禁填充 `rpm_per_window` 或 `rpm_provenance` 字段；不为设备虚构测速能力。

## 4. 交付接口与验收

文件：`phm_design.industrial_contracts.physical_graph_design`：

- `propose_physical_graph_study(equipment, train_manifest, trial)`：冻结元数据条件允许的 A/B/C/D 图与 R96 子计划，**无训练，无波形扫描，无 VAL/FINAL 访问**；
- `generate_graph_source_features(study, equipment, train_manifest, trial, raw_records)`：校验原始 SOURCE 观测并生成全部图对应的**一记录一特征行**，附测速数值哈希和未认证声明；
- `evaluate_physical_graph_study(..., provider=...)`：调用已有 R96 fold-local pipeline / R94 provider，计算每种结构对应的**SOURCE 训练集内部**评估收据，不改算法库。

跨平台 CI 将运行 Linux 3.11/3.12、Windows 3.11 的 R93–R98 回归，并进行实际 sklearn LogisticRegression、DecisionTreeClassifier 的拟合、包络 200Hz 调制数值检验、3,000rpm 正确阶次映射、拒绝 12kHz 错钟/无转速/变速/目标字段/错误配额/篡改图等测试。**测试数据为明确的合成波形，不是实测轴承故障泛化证明**。

安装测试依赖：

```powershell
cd source_product
python -m pip install numpy scipy scikit-learn pytest
$env:PYTHONPATH="src"
python -m pytest -q tests/phm_design/test_r98_physical_graph_design.py
```

R98 没有使用/分发第三方受限模型权重，也不需要本机真实工业设备，完整 CI 会运行完整研究栈回归。

## 5. 资格边界和下一研发任务

- `independent_diagnostic_generalization_proven=false`：SOURCE 留组不是独立外部设备验证，且 R95–R98 反复根据测试探索可能造成选择偏差；
- `physical_clock_tachometer_and_units_independently_verified=false`：`rpm_per_window` 虽严格从独立测速通道索取，但目前由输入者**自述来源**，未独立认证；
- `recording_level_fusion_is_sample_synchronized=false`：标量同记录拼接没有逐点同步的证据；
- `cpu_gpu_compute_cost_parity_proven=false`：相同 trial/fit 数量不等于相同 GPU、时间、能耗；
- 不对 CWRU 原始两文件错误声称有 tachometer；R97 真实 CWRU 测量只证明没有测速前提也能做时域/包络类描述，**绝不自动开启阶次图**；
- Stable main 和已有针对自组织模糊机制的保留/否定证据不变，不能将本轮 R98 结构可运行作为自组织泛化优势证据。

**下一项建议的核心研究是：可解释的“为什么产生某一设计”的机理/可观测性检查与因果证据、独立真实设备任务的严格外部评审、成本受限设计搜索；不能只继续堆叠更多 FFT 特征。**
