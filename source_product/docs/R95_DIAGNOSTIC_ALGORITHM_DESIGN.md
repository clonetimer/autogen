# R95：autogen 的核心工作——自动设计诊断算法

**2026-10-25 · Draft / 研究实现，不是工业跨设备资格。** R95 基于 R94（P1.2），R94 基于 R93（P1.1）。三个增量目前为 Draft，未合并 Stable `main`。

## 为什么开发 R95

`autogen` 不是算法库、AAS 的子进程运行器、也不是单纯的数据连接器。它应对设备、任务、观测和证据约束回答：

1. **提出什么算法**，为什么适用于任务？
2. **使用哪些可观测传感器及哪种特征组合**，哪些禁止的标签、源真值或未来字段不得使用？
3. **超参数与模型复杂度**怎样在声明的试验预算内变化？
4. **按哪些真实物理设备或运行环境分组**形成仅限 SOURCE TRAIN 的内部比较？
5. 有足够独立设备与故障样本吗？如果没有，**拒绝排名和宣称优势**。
6. 最终只返回**训练集内部探索性短名单**，真正的盲法 VALIDATION/FINAL 必须由外部评审继续完成。

`torch_like`/AAS 提供已实现的算法对象、训练、判定、保存和进程管理。R95 不重写逻辑回归、树、随机森林、IsolationForest、TSK 等模型数学。

## 当前实际实现

新增 `phm_design.industrial_contracts.design_engine`：

- `propose_diagnostic_study(equipment, source_train_manifest, trial)`：读取 R93 合同，**只接受 TRAIN**；按 `max_trials` 生成确定性 `algorithm_family × sensor_subset × approved_hyperparameter` 组合，并冻结 SHA-256 试验清单。只有 R94 真正映射的四种 provider 才可入选（classification: linear/tree/random_forest；normal-only anomaly: isolation_forest）。
- 传感器搜索依次尝试全合法输入、每个单通道、可行的两通道组合。首轮保证不同算法家族使用**相同完整观测集**，其后探索删减传感器。没有读取训练标签做隐藏特征选择，不生成不明来源的故障真值特征。
- 超参数搜索严格白名单：LogisticRegression `C∈{1,0.1}`、DecisionTree `max_depth∈{5,3}`、RandomForest/IsolationForest `n_estimators∈{40,20}`。随机种子来自 TrialSpec；算法数学由现有提供者执行，不允许任意 `kwargs`、任意动态类。
- 物理设备/工况留组折数源自**TRAIN 分组身份**，训练与内部评分记录完全不相交，不把同一个轴承的窗口当独立轴承。分类折需保留两类训练覆盖；跨环境实验要求至少 3 个 SOURCE TRAIN 环境。
- `execute_source_design_study(frozen, ...)`：拒绝修改过的模型候选、源 SHA、任务、种子和折；按固定源留组折创建新 `TorchLikeTabularCandidate`，拟合只见该折训练的设备和源标签，`predict` 只接收允许的传感器数值；独立的 SOURCE TRAIN 内部审阅计算组宏准确率、最差组准确率、每类召回率与健康误报比例。
- 在所有候选**相同内部留组折**下，提供确定性临时排名。记录总拟合次数和实际训练壁钟，而不是凭相同 `max_trials` 假定训练开销相同。
- 支持 SOURCE 组样本不足、仅正常异常检测、缺失可执行算法时明确 `ABSTAIN_SOURCE_EVIDENCE_INSUFFICIENT`；绝不通过合成故障标签制造排名。
- 硬限制一次实验最多 **64 次 SOURCE 拟合调用**（强制在任何执行前做静态检查）；训练墙钟仅事后记录/检查，**仍未实现进程级抢占与公平 FLOPs/峰值内存核算**。
- `design_cli` 支持 `propose` 与 `source-study` 分开运行；后者必须有之前归档的完整研究清单 SHA，原报告禁止覆盖；整个命令不存在独立 `VALIDATION`、`FINAL` 参数。

### 非算法研发的功能边界

仍由 R93 管合法数据元信息/分组；R94/AAS/torch_like 管算法实际训练及判决。R95 执行策略是**结构化、可复现的有限设计空间搜索**，并非无限制神经网络结构发现；神经网络与模糊规则现在只属于有文档的待对接候选，不会偷偷运行其他模型来冒充。

## Windows/Linux 使用

从 `source_product` 目录运行；需要 Python 3.11+。生成候选阶段只用 Python 标准库；真正执行阶段需要本机已可信安装并审核的 `torch_like` 依赖：

PowerShell:

```powershell
$env:PYTHONPATH="src"
python -m phm_design.industrial_contracts.design_cli propose `
  --equipment examples/industrial_contracts/bearing_equipment.json `
  --train-manifest examples/industrial_contracts/bearing_train_only.json `
  --trial examples/industrial_contracts/bearing_trial.json `
  --output source-design.json
```

审阅并**单独归档** `source-design.json` 和 `study_sha256`。然后在已有可信算法提供者的环境中：

```powershell
python -m phm_design.industrial_contracts.design_cli source-study `
  --equipment examples/industrial_contracts/bearing_equipment.json `
  --train-manifest examples/industrial_contracts/bearing_train_only.json `
  --trial examples/industrial_contracts/bearing_trial.json `
  --source-features examples/industrial_contracts/bearing_train_features.json `
  --frozen-study source-design.json `
  --expected-study-sha256 "<归档的64位SHA256>" `
  --output train-internal-comparison.json
```

压缩机 TRAIN 示例目前只有两条**正常**样本，其 `propose` 可以产生 IsolationForest 候选，但 `source-study` 必须拒绝报告故障效果排名。不能当作 P2 真实 MetroPT 设备结果。

## CI 证明了什么

- 合成元数据、组身份与源数据测试验证：候选生成重复性、实验 SHA、试验数限制、物理组和源标签隔离、恶意元数据拒绝、工况支持检查、没有故障样本时自动拒选。
- 使用**实际 sklearn LogisticRegression/DecisionTreeClassifier** 做小规模源留组训练与比较。算法真实训练，样本仍是合成技术夹具：这不是实测工业诊断质量，更不是安装自有真实工业设备的要求。
- GitHub CI 通过一个实现 `torch_like` API 的 sklearn 兼容测试提供者运行。**不能说已在这项 CI 中检出用户私有 `torch_like` 仓库**；真实跨仓库 token 控制的测试仍按 R94 文档要求运行。
- Linux Python 3.11、3.12、Windows Python 3.11 的 R93/R94/R95 联合专项回归。CI PASS 仅证明工程功能，不代表跨设备独立效用。

## 当前技术结论与后续

R95 首次把 `autogen` 的**自动提出诊断设计方案 + 有边界地比较**落实到执行层。但它尚不能独立证明一组算法可推广至另一台工业设备，也不能自动生成复杂新网络架构。

下一阶段优先：

1. **增强设计语言**：可验证的归一化、缺失处理、窗口特征、采样率一致性、模型组合/集成、物理先验约束，不应靠遍历几种固定模型代替自动设计。
2. **更强的 SOURCE 选择协议**：多种预注册工况/物理批次、分层组交叉验证和多种种子，训练与验证预算严格对齐；拒绝把源搜索 OOF 当最终性能。
3. **真实 P2/P3**：物理设备独立身份的旋转机数据、时间/事件隔离的 MetroPT 压缩机数据；单设备数据只能支持单设备时序异常研究。
4. **组合诊断**：复用 `torch_like.automl` 与已有算法/健康评估流程，候选生成器选择和组合现有算法，而不是重复实现它们。
5. **独立评审制品**：冻结拟合流程后，由受信任的验证端读独立 VALIDATION/FINAL，补齐成本/误报/任务目标及适用边界报告，再评估是否合并 Stable。

目前 R95 的 `provisional_source_shortlist` 必须始终标明 **SOURCE TRAIN INTERNAL、NOT VALIDATED、NOT PRODUCTION**，不允许把内部排名自动升级为可交付工业算法。
