# R93 / P1 — 跨工业设备最小数据与试验合同（独立功能增量）

**2026-10-25 · 状态：Draft / 研究设计合同，非 Stable 新资格。**
**分支从 `main` 单独创建；不依赖 R80–R91 的层层 Draft PR。**

## 1. 本轮完成什么、未完成什么

实现纯 Python 标准库、无 Torch/NumPy/Basilisk 运行依赖的三个可验证合同：

- `EquipmentCard`：设备家族、时间字段、可观测的传感器/控制量、模态、工程单位、采样频率及决策时刻可见性；
- `DatasetCard`：测量/原生物理仿真/合成数据来源声明、许可、物理设备身份、采集会话、工况、TRAIN/CALIBRATION/VALIDATION/FINAL、独立源文件 SHA、UTC 数据区间；
- `TrialSpec`：任务类别、标签字典、推理输入字段、候选算法家族、实验范围（新设备/新工况/同设备未来）、资源预算和种子。

**注意：** 以上均是元数据合同；不读取工业原始数据文件，不真实核验设备身份，不自动下载或训练模型，不赋予可信部署资格。SHA 是用户提供的自述值，**不是**对实际数据字节的核验。后续 P2 负责真实设备接入与字节核验，P3 才进行算法自动选择与评估。

## 2. 已提供两个独立工业示例

`source_product/examples/industrial_contracts/`

| 设备示例 | 任务 | 数据划分和目标 | 保护措施 |
|---|---|---|---|
| bearing | 已标注正常/故障分类 | `physical_group` / `unseen_physical_device` | 同一轴承不得在 TRAIN 与 VAL 等不同角色出现；相同原始文件不能另编物理 ID |
| compressor | 正常数据学习与异常检测 | `temporal_forward` / `future_same_device` | 同一设备训练时间必须早于验证，角色之间不少于声明的 3600 秒 embargo，原始文件或会话不能跨角色重复 |

示例的 `origin="synthetic"`、占位 SHA 和传感器单位均明确说明：**仅用于合同功能测试，不是 CWRU、MetroPT 或 Paderborn 的实测证据。** 真正的单台压缩机多时间段不等于多设备泛化试验。

### 真实项目使用方式

工作目录为 `source_product`。只需要 Python 3.11+，核心合同不要求 GPU 或额外 ML 包。

PowerShell：

```powershell
$env:PYTHONPATH = "src"
python -m phm_design.industrial_contracts source-plan `
  --equipment examples/industrial_contracts/bearing_equipment.json `
  --dataset examples/industrial_contracts/bearing_train_only.json `
  --trial examples/industrial_contracts/bearing_trial.json `
  --output bearing_source_plan.json
```

Linux/Git Bash：

```bash
export PYTHONPATH=src
python -m phm_design.industrial_contracts source-plan \
  --equipment examples/industrial_contracts/compressor_equipment.json \
  --dataset examples/industrial_contracts/compressor_train_only.json \
  --trial examples/industrial_contracts/compressor_trial.json \
  --output compressor_source_plan.json
```

只有受信任数据管理员可运行包含 CALIBRATION/VALIDATION/FINAL 的**全清单审计**：

```shell
python -m phm_design.industrial_contracts audit --equipment examples/industrial_contracts/bearing_equipment.json --dataset examples/industrial_contracts/bearing_trusted_audit.json --trial examples/industrial_contracts/bearing_trial.json --output bearing_trusted_audit.json
```

`source-plan` **拒绝**任何 CALIBRATION/VALIDATION/FINAL 行；不能将完整清单直接提交给设计工作线程。输出只有 TRAIN 基于的候选名、允许观测列、源组和环境数量、预算以及合同 SHA，没有验证标签、完整数据值或指标。既不会执行训练也不会下载数据。输出文件不允许覆盖。

## 3. P1 具体边界和失败策略

1. **设备与传感器**：sensor 名必须合法、唯一；明确保留 ground_truth、fault_label、failure_time、true_rul、role、physical_id 等禁止字段；未在决策时可用的通道不得作为部署特征。工程单位可以标注 `UNVERIFIED`，但后续不得据此主张完成仪器标定。
2. **物理隔离**：新设备/新工况实验必须按真实物理对象分组；同一原始文件字节 SHA 不得冒充不同设备，同一物理对象不能跨 TRAIN/VAL/FINAL 等角色。
3. **时序隔离**：同一设备预测未来任务使用 `temporal_forward`，不同角色之间要求正的 embargo 秒数，不允许时间区间重叠或在验证后重新放入 TRAIN；同一采集会话和原始文件不得跨角色拆分，即使窗口不重叠。
4. **任务限制**：本轮支持分类与仅正常训练的异常检测定义。RUL 的维修事件、删失、失效时间定义尚未实现：不得把分类任务简单改名为 RUL。
5. **资源预算**：包括候选数、CPU 壁钟训练预算（秒）、内存上限（MB）、随机种子；这里只核验规范，不执行硬资源停止或计算量相等。
6. **输入输出隔离**：训练的标签只允许用作监督真值元数据，不能作为部署特征；源设计 plan 不可读取受保护角色。
7. **拒绝而非猜测**：字段缺失、未知设备族、非法标签、角色不符、时间非 UTC、缺失采样单位或来源、互相矛盾的独立泛化声明，一律拒绝。
8. **不继承既有证书**：不改变 `main` 当前 0.7.0 限定卫星仿真离线资格。

这些规则不会解决来自原始设备厂商的伪造/错误物理 ID；无法通过软件对自述来源施加真实性保证。R89、R91 的真实字节验证经验后续可供 P2 吸收。

## 4. 与未来架构对接

- **上游**：设备/数据导入可来自 AAS、CSV、Parquet 或第三方采集系统；P1 不强制使用一个特定数据库，也不声称已实现 AAS 集成。
- **候选算法**：只登记家族（常数、统计、线性、树、随机森林、CNN、TCN、GRU、可选 TSK 模糊）；具体 `fit/predict/export/resource_receipt` 适配与 `torch_like` 集成留到 P3/P4。
- **下游**：`ModelCard`、`EvidenceReceipt`、可解释适用范围与 ONNX/CPU 推理包在后续单独受控实现。
- **测试**：Linux 3.11、Linux 3.12、Windows 3.11 的 pytest 和真正的 CLI 源计划输出。CI 绿灯=合同软件可用，不代表设备/算法已真实验证。

## 5. 后续按顺序推进，不再堆叠架构搜索

**P1.1**（本 PR）：两类设备字段与时间/物理隔离合同，生成 TRAIN-only source plan；测试通过后独立评审。

**P1.2**：将 R80/R81 的实用执行接口以小 PR 形式适配合同，固定最小统一候选接口，不直接把长链研发 PR 合并。

**P2**：优先真实 Paderborn 或 CWRU 转子振动测量与 MetroPT 压缩机多变量实际运行数据；记录源文件 SHA/单位、事件定义、同实体/同机器数据限制、许可与受限泛化范围。严禁把切窗后的数量当作独立设备数。

**P3**：用同样的真实设备和时序组划分，匹配数据与成本预算，比较统计/传统 ML/NN，TSK 只作为可选探索候选，允许明确的负结论；记录推理时延、拟合次数与内存，进行真正的独立或前瞻性验证。

**P4/P5**：模型制品 + 证据报告，再做 Windows/Linux 非程序员交互向导和可选工业只读适配。

实际是否可以合并 Stable 仍需单独执行主干全量回归，不能把当前专项 CI 当成全量发行验收。
