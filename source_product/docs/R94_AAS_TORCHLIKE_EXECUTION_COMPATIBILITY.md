# R94 / P1.2 — 复用 AAS 检测器合同与 torch_like 算法库（非重造算法）

**状态：2026-10-25 Draft；基线 R93 P1.1。** 原有 `main` 的 phm-design 0.7.0 限定卫星仿真 Stable 范围不改变。

## 上传项目审查与来源

审查了用户重新上传的 `20261023_3(1).zip`：其实际根目录为 **AAS/**，不是 `torch_like` 算法库的源码。

- ZIP SHA-256：`0fc9526266da4b56abe65bc2593feda6c23178e6cd0e97d744fa9465b566c0fd`。
- `AAS/detectors/base.py` SHA-256：`996e0b7c16490433ea33d6ed1ae41adae2379ab269ab71c0f470da67aa52cbe3`。
- `AAS/detectors/loader.py` SHA-256：`51f037cbd89550a5d7e49a6b5b0709ba949933e0df56df394e9e99ea298252de`。
- `AAS/detectors/decision.py` SHA-256：`1cf7df0364860bd8c1fd6370633b664d3f1036b45d37d558e707826a87574c7a`。

**用户上传源码未复制、嵌入、推送到 autogen 仓库。** R94 仅抽取协议边界、编写适配器及测试用替身；是否应与 AAS 形成产品级依赖仍需双方版本钉住与兼容性验收。

真实上传 AAS 中观察到：

| 原有代码 | 实际接口 | R94 处理方式 |
|---|---|---|
| `detectors.base.BaseDetector` | `fit(train_X, train_y=None, valid_X=None, valid_y=None)` | 只传 R93 TRAIN 源观测与可选 TRAIN 分类标签 |
| `BaseDetector` | `decision_mode=score_threshold/native/label` | 不替换、不二次重算最终标签 |
| `BaseDetector.detect` | `pred_label`, `anomaly_score` 可为 null，`threshold(s)`, `decision_mode` | 保留原生结果，拒绝伪造分数/标签 |
| `detectors.loader.detect_with_detector` | `preserve_state=True` | 可由 AAS 主程序注入权威入口，保证原有状态快照/恢复 |
| `detectors.registry` | YAML、`class_path`、依赖、判定模式、config schema | AAS 注册和加载；autogen 仅保留元数据与选定候选家族绑定 |
| `scripts/server/algorithm_process.py` / `algorithm_worker.py` | 子进程任务监督、超时、并发和限制 | 保留 AAS 宿主运行职责；R94 **没有**实现硬进程隔离 |
| Phase 39–41 | 模型产物、模型注册、独立评估和进程隔离 | `autogen.export_model()` 只返回**交接合同**，不直接生成可执行模型或不可信 pickle |
| `torch_like` 已连接 GitHub 仓库 | 算法注册、ModelSpec、ModelRuntime、AutoML、save/load | 由 `torch_like` 提供实际 ML/NN 后端，不将其模型实现复制进 autogen |

**分层职责：**

```text
autogen — EquipmentCard/DatasetCard/TrialSpec、候选搜索、TRAIN-only源约束、
          预算/证据/公平比较、最终算法设计方案
    |
    v
AAS     — 检测器描述符/注册/算法调用、判定模式、阈值/标签唯一权威、
          模型存取/发布、带状态恢复的独立评估、进程工作者
    |
    v
torch_like / 其他已注册算法后端 — 具体统计/ML/NN/健康评估/RUL 实现
```

上述是**推荐职责结构**，不表示 autogen 已集成所有 AAS/torch_like 具体算法与生产进程；本 PR 只实现 AAS 薄桥。

## 交付实现

`phm_design.industrial_contracts.aas_bridge`：

- `bind_existing_aas_detector(detector, aas_descriptor, candidate_family=..., detection_service=...)`
- `AASDetectorBridge.fit(equipment, train_manifest, trial, train_features)`
- `AASDetectorBridge.predict(observations)`
- `AASDetectorBridge.export_model()`（只输出描述信息：**模型字节由 AAS 保存**）
- `AASDetectorBridge.resource_receipt()`

**允许的模型执行范围仅为 R93 当前支持的二分类和单模型正常样本异常检测**。分类器的 AAS `label` 需约定 `0=healthy`、`1=fault`；异常检测的 `score_threshold`、`native` 仍由算法维护自身状态。多类别标签、未标定分类概率、RUL、领域无关的工业部署资格等不在 P1.2 合同内。

实例由 AAS 的原始 `load_detector` 创建；autogen 不按 `class_path` 导入任意 Python 代码。每个实例只允许 **一次 TRAIN 拟合尝试**，不接受验证/最终标签。按 R93 的 `input_channels` 构造数值矩阵；未声明字段、真假标签、缺失、重复 record_id 或非数值输入都拒绝。所有推理调用使用 AAS 权威 `detect()` 的结果，不用 score 重新生成标签；score 仅存在于算法实际提供的模式中。

**避免测试集阈值泄漏：** `score_threshold` 模式必须具备 AAS `needs_threshold_calibration`、`fit_threshold`、`score`；需要校准时只使用 TRAIN 得分完成校准，验证前若仍未校准即拒绝。支持以 `AAS.detectors.loader.detect_with_detector` 作为 `detection_service` 注入，自动 `preserve_state=True`；否则调用原 `detect` 并使用 `export_evaluation_state`/`restore_evaluation_state` 包裹。

**资源记录局限：** 本桥记录来源组数、训练调用次数、训练壁钟时长，但未实现强制终止、峰值内存追踪、模型字节度量或独立工况性能计算。不能凭此声称满足 P3 的公平计算预算。

## AAS 原始上传文件本地兼容性实测（不作为 GitHub CI 的一部分）

本地直接从上述 ZIP 读取 AAS 源码，并调用了原始 `detectors.loader.detect_with_detector(..., preserve_state=True)`：

| 检测器 | 原始数据 | 原生执行结果 | 审计 |
|---|---|---|---|
| `ThreeSigmaDetector` | 小型合成双行 3 通道输入 | `score_threshold`; [0, 1] | 训练和最终标签成功 |
| `IsolationForestDetector`（native） | 同上 | `native`; [0, 0] | 原生算法自己决定；不应更改为期望的 [0,1] |
| `BaseDetector` 子类 label-only | 合成轴承二分类四条 TRAIN | `label`; [0, 1]，无 score/threshold | 不生成虚假的异常分数 |

**发现一处真实 AAS 状态细节：** 对 `ThreeSigmaDetector` 执行一次带 `preserve_state=True` 的检测后，其 `export_evaluation_state().threshold_state.config` 会补写 `threshold_method=fixed`。阈值及判定不变，原始完整状态字典并不逐字节相等。这是被审查项目当前行为，不能宣称模型状态 bitwise identical。R94 不自行修改 AAS 代码，建议单独维护 AAS 回归测试及修复。

## 测试/使用示例（宿主已安装 AAS）

```python
from detectors.loader import load_detector, detect_with_detector
from phm_design.industrial_contracts.aas_bridge import bind_existing_aas_detector

aas_yaml_descriptor = {
    "name": "three_sigma",
    "class_path": "detectors.statistical_detectors.ThreeSigmaDetector",
    "config": {},
    "decision_mode": "score_threshold",
}
detector = load_detector(aas_yaml_descriptor)  # AAS 原有入口
bridge = bind_existing_aas_detector(
    detector,
    aas_yaml_descriptor,
    candidate_family="statistical",
    detection_service=detect_with_detector,
)

# equipment, train_manifest, trial 使用 R93 JSON 合同；
# train_features 仅为 TRAIN ID / 合法观测通道数值。
bridge.fit(equipment, train_manifest, trial, train_features)
result = bridge.predict(unlabeled_sensor_rows)
handoff = bridge.export_model()
receipt = bridge.resource_receipt()
# 真实保存/加载交给 AAS 模型库；handoff 不包含 Python pickle。
```

## CI 和后续工作

三平台 CI 会运行 R93 合同回归和 R94 的 AAS 契约替身，包含 score_threshold/native/label、重复训练禁止、TRAIN-only 校准、输入身份与标签泄漏、状态恢复、异常输出等。**由于用户 AAS ZIP 未提交到 GitHub，这些 GitHub CI 不能冒称运行过 ZIP 的原始 AAS 源码**；原始上传代码的验证如上单独陈述。

**P1.2 之后：** 接入受版本约束的 AAS 运行者并补上带真实 AAS 文件的可复验跨平台兼容测试；复用 `torch_like` 模型目录/算法注册，而非新造序列化。P2 再接真实工业设备数据；P3 做物理设备独立、公平预算与盲测性能对照。

本 PR 无模型质量提升主张、无生产资格升级、无主干合并。
