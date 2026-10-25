# R96：面向跨领域诊断的自动流程设计（P1.3，研究版本）

**状态：Draft，基于 R95 / R94 / R93 研究链；未改变 main 的受限卫星仿真 Stable 资格。**

## 研究目标：让 autogen 设计诊断流程，而不是制造另一套算法

R95 已经搜索“算法类型 × 合法传感器子集 × 超参数”。R96 将“预处理”纳入真正的**可执行结构搜索**，在同一批 SOURCE TRAIN 物理留组折上比较：

```text
EquipmentCard + DatasetCard + TrialSpec（R93）
      -> [传感器选择] -> [折内训练的预处理]
      -> [torch_like 已有分类/异常估计算法] -> [源训练集留组评估]
      -> 仅训练集内部候选短名单 / 证据不足拒选
```

并未自研 LogisticRegression/DecisionTree/RandomForest/IsolationForest 的算法数学；完整 AAS 判决及报警后处理仍由 AAS 管理，而不是 autogen 强加。

## 实际交付

`phm_design.industrial_contracts.pipeline_design` 提供：

- `propose_pipeline_study(equipment, source_train_manifest, trial)`：无训练、无验证数据访问；纯粹依赖 R93 元数据和 R95 候选计划生成、冻结 SHA-256 的**预声明流水线**。
- `fit_preprocessor(transform, channels, train_matrix)`：在**当前 fold TRAIN** 上估计处理器状态，支持 `identity`、`standard_scale`、`robust_scale`、`signed_log1p`；最后一种可避免直接对正负振动/压力使用不合适的无符号对数。
- `apply_preprocessor(state, matrix)`：使用 SHA-256 锁定的参数转换推理数据；禁止任意代码执行、非法变换、缺失列、NaN/Inf 和状态篡改。
- `evaluate_pipeline_study(frozen, equipment, source_train, trial, features, provider=...)`：每个设计/物理留组折**重新拟合预处理器，再训练独立算法实例**；留出组只做 transform + predict，绝不参加均值/MAD/标准差估计和模型训练。记录每个折的训练记录数量、预处理参数散列、保留组散列、提供者拟合调用次数、时间和源数据组级指标。
- 优先比较所有可执行算法在**相同完整传感器**和 `identity` 下的结果，再尝试其他固定预处理方案；需要更多试验预算时再加入 R95 的传感器和超参数变种。所有候选依然受 R93 `max_trials` 和最多 64 次源拟合硬边界限制。
- `robust_scale` 使用 TRAIN-only 中位数/MAD；零方差通道采用固定的有界数值替代，不能通过把验证数据补入拟合来掩盖数据质量问题。

## 数据分离与结论边界

这里的 SOURCE group OOF 分数**用于选择设计探索方向**。不能把在同一来源上反复迭代所得短名单称为独立验证结果；不能把同轴承各窗口当作独立设备；仅有正常样本的 MetroPT 风格异常任务仍明确拒绝产生故障效用排名。

`source_data_independent_of_prior_design_iterations=false`、`independent_model_validated=false`、`production_qualified=false`。没有声称某一种标准化总是更好，也没有声称通过额外处理能证明模糊神经模型的泛化优势。

## 如何验证

```powershell
cd source_product
python -m pip install numpy scipy scikit-learn pytest
$env:PYTHONPATH="src"
python -m pytest -q tests/phm_design/test_r96_diagnostic_pipeline_design.py
```

CI 在 Linux Python 3.11/3.12 和 Windows Python 3.11 执行 R93/R94/R95/R96 联合测试，使用实际 sklearn 模型训练，但**测试数据仍为合成合同夹具**。可选私有 `torch_like` 包的端到端导出重放如未安装会被跳过；不能把接口替身通过说成真实 `torch_like` Release 已验证。

## 尚未覆盖的诊断算法自动设计能力

- 分时段的滑动窗口、频谱/包络、工况归一化和多传感器时间对齐，均需要可靠采样率、源时钟和运行工况确认，不能仅对固定行数据随意推断时序。
- 受约束的算法组合/集成、可解释逻辑、半监督、新型神经网络结构、RUL 生命周期预测属于后续模块，必须复用成熟算法/执行接口。
- 内部 OOF 既不具备独立外部验证地位，也未严格匹配 FLOPs/GPU/峰值内存；当前记录 fit 调用和拟合秒数，不得宣称公平成本已完全保证。
- 生产级进程限时、内存配额、可信模型加载、AAS worker 和外部验证审计还未由 R96 完成。

**验收标准是设计机制与数据隔离可运行，而非立刻提升算法准确率。** 后续依次推进物理采样特征设计、可复现实测工业数据试验、严格成本对照与外部验证，允许负面结果。
