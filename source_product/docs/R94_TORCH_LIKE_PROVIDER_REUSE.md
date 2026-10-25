# R94 / P1.2 — 对接已有 torch_like 统一算法接口，不重复实现

**状态：研究 PR / 尚未合入 main；仅执行 P1.2 的窄范围适配。**

## 补充：重新上传的 AAS 项目已完成接口审查

用户重新上传的 `20261023_3(1).zip` **实际是 AAS 工程**，不是
`torch_like` 单独源码。R94 因此同时提供两个**可替代而非嵌套强制依赖**
的薄适配器：`torch_like_bridge` 面向直接使用已注册的 native sklearn
候选；`aas_bridge` 面向运行在已有 AAS 宿主中的 `BaseDetector` 插件，
保留其 `score_threshold` / `native` / `label` 最终判定权、
`detect_with_detector(preserve_state=True)` 和现有模型注册/保存。

详见 [R94 AAS 原始上传包接口及本机兼容性审计](R94_AAS_TORCHLIKE_EXECUTION_COMPATIBILITY.md)。
AAS 原始源码没有上传到本仓库。该上传包所含检测器可内部调用
`torch_like`，但 `autogen` 不需要再调用一次相同的判决或阈值处理，
否则会发生双重判定。

## 先复用哪些能力

| 需求 | 提供者 | autogen 仅负责 |
|---|---|---|
| 模型能力发现 | `torch_like.classical.get_estimator_info`、`ModelSpec` | 与 R93 TrialSpec 的任务、算法家族和输入可用性核对 |
| 分类算法及异常算法 | `create_estimator` → sklearn 原生对象 | 读取 TRAIN-only 源特征；任务、物理分组与样本隔离 |
| 拟合和预测 | `as_runtime(model).fit/predict` | 验证任务/标签及只传入约定传感器 |
| 异常判别 | IsolationForest 原生 `predict` / `decision_function` | 约定异常分数为 `-decision_function`，不修改训练时阈值 |
| 模型持久化 | `torch_like.save_model` | 新文件目录、SHA 审计、资源状态报告 |
| 可信模型重载 | `torch_like.load_model` | **仅信任本机生成且来源经过验证的模型** |
| 多算法搜索、跨组 CV | `torch_like.automl` Backend/CandidateSpec/ProtocolAudit | 后续 P3 统一预算、外层评价与最终资格判断 |

### 明确不复用错误功能

- `ModelRuntime` 并不意味着任何模型都天然支持 `fit`；
  sklearn 适配器和 PyTorch 训练器依然有不同生命周期。
- `torch_like.serialization` 的 sklearn 制品使用 **joblib/pickle**，
  因此 P1.2 只允许操作受信任的本机导出模型。不能直接加载用户上传的
  未审核 joblib 文件。
- `SignalBatch` 用于原始序列数据；R93 示例是**每个源记录一个 tabular 特征向量**，
  不应把没有正确时钟/单位的离散压缩机记录伪装成均匀采样序列。
- 不在 `autogen` 内重新实现 sklearn/torch_like 算法数学、拟合过程或持久化格式。

## R94 新增薄适配代码

`phm_design.industrial_contracts.torch_like_bridge` 提供
`TorchLikeTabularCandidate`，方法：

```python
from phm_design.industrial_contracts.torch_like_bridge import TorchLikeTabularCandidate

candidate = TorchLikeTabularCandidate("linear")
candidate.fit(equipment, train_only_dataset, trial, source_features)
predictions = candidate.predict(unlabeled_sensor_features)
evidence = candidate.resource_receipt()
model_manifest = candidate.export_model("local-trusted-model-directory")
```

四个确实存在的原生提供者候选：

| R93 任务 | autogen 家族名 | torch_like 原生 sklearn 对象 |
|---|---|---|
| 二分类 | `linear` | `LogisticRegression` |
| 二分类 | `tree` | `DecisionTreeClassifier` |
| 二分类 | `random_forest` | `RandomForestClassifier` |
| 正常基线异常检测 | `isolation_forest` | `IsolationForest` |

`torch_like` 已经内置的随机森林、异常检测器、线性分类器没有复制到
`autogen`。R93 已声明的 `cnn1d` / `tcn` / `gru` 等候选只是**注册级别**，
不表示已有合适的序列数据适配和训练配置。名为 `statistical` 的研究候选
也不自动替换为 IsolationForest：后者是独立算法，避免偷换概念。

## 无数据泄漏门禁

P1.2 开始执行前必须先通过 R93 的 `compile_source_plan`：只允许 TRAIN，
保护角色、同设备/跨工况和时间跨度由 R93 原有合同处理。P1.2 验证每个
`record_id` 与用户提交的源特征精确对齐。训练标签仅用于监督分类，
从不传入 `predict`；执行器拒绝任何新增的 `role`、`label`、
`ground_truth`、`physical_id`、NaN、Inf 及非预声明字段。

资源报告仅包含**本次实测训练墙钟时间、源记录数、源物理组数、拟合次数**。
这里尚未实现训练进程的硬抢占、峰值内存/FLOPs 测量和跨算法资源均衡。
因此报表显式标注对应限制，不将同 `max_fit_seconds` 声称为计算成本匹配。

## 验证条件与后续

- 无 provider 的 Linux 3.11/3.12、Windows 3.11：使用受控 spy 适配器
  验证**路由、权限、源观察一致性与失败隔离**，不把 spy 当作真的模型。
- **真实库集成测试为单独手动验收，不算自动 PR 通过项。**
  `torch_like` 的 GitHub 仓库不允许由 `autogen` 的默认 Actions
  `GITHUB_TOKEN` 直接跨库检出（首次 CI 返回 `Repository not found`）。
  工作流只在 `workflow_dispatch` 时运行这一受限测试，
  必须先由维护者在 `autogen` 仓库 Secrets 配置具有
  `clonetimer/torch_like` **Contents: read** 权限的
  `TORCH_LIKE_READ_TOKEN`，然后手动运行 R94 工作流。
  测试锁定提交 `1e603915896e33fd9f326073c42a477718166b68`
  并实际执行四种原生 sklearn 模型的 ModelRuntime 和可信制品保存/重载。
  **在这条实际测试 SUCCESS 前，不能宣称真实库已完成跨仓库集成验收。**
  请勿把私有 token 加到源代码、PR 正文或日志中。
- 所有示例源数值是合成测试数据，不是公开 CWRU/Paderborn/MetroPT
  的真实独立诊断结果。模型不获得 Stable/industrial 资格。
- 下一阶段 P2 应接入实际工业测量时保留物理样本身份、文件 SHA 和观测
  单位。P3 应复用 `torch_like.automl` 的任务、候选/runner 与
  协议审计，**不要**再创建另一套 AutoML 搜索引擎。

> 本轮已检查实际上传 AAS 源码。当前只有两个受控适配方向，分别是
> 直接 torch_like 算法提供者，以及 AAS 中原有的检测器。两套接口都
> 不替代 R93 分组审计、P2 真实测量数据核验或 P3 独立模型性能验证。
