# PHM AutoDesign 0.8.0rc1 — Stable 收敛 / 发布范围与验收门禁

**状态：2026-10-25 正式启动软件 Stable 候选阶段；当前为 PRE-RELEASE（未正式发布/未创建 GitHub Release）。**

## 1. 为什么需要收敛

已合并至 `main` 的 R93–R99 研究能力和原有 0.7.0 仿真算法共存，完整 Python 集成测试在合并前达到 **848 passed、47 skipped**。这证明了显著的工程成熟度，**不能**替代独立干净环境安装、真正已连接算法库的 E2E 和发布包一致性证明。本次冻结大规模新增模型、物理特征、搜索结构的研发，优先完成交付可重复性。

**既有资格不可升级**：`0.7.0 Stable` 的受限 Basilisk 15 场景离线诊断资格与回滚制品保持原样。新 `0.8.0rc1` 先作为框架软件候选；本轮不产生新的工业设备或飞行安全认证，也不再用“必须证明新算法优于固定基线”作为软件框架的普遍发布前置条件。对外成果按“框架稳定”“特定算法/设备合格”“研究性功能”分别声明。

## 2. 冻结的软件支持边界

| 工作范围 | 0.8.0rc1 状态与门禁 |
|---|---|
| 核心 `phm-design` 算法设计、R93 数据合同/角色隔离、R95 SOURCE-only 诊断候选计划、R96 预处理和 R97/R98 信号流程 | 框架候选软件；要求安装、可重复运行和拒绝非法输入 |
| R94 `torch_like` 四类表格候选的 native sklearn 训练、预测、保存、恢复 | 发布阻断项：**真实 `torch_like`、精确 commit、全新 wheel 安装、Linux/Windows**，不能用 spy 替代 |
| R94 `AASDetectorBridge`、AAS 插件 YAML/阈值/worker 深度耦合 | **可选实验性集成**；用户上传的 AAS 源码不可自动作为已安装系统依赖，必须另行验收才能宣称一键集成 |
| R98 阶次/包络/融合和 R99 设计解释、成本研究与实测门禁 | 研究性 opt-in 功能；未取得工业通用效果资格 |
| 已发布 0.7.0 Stable 受限 15 场景模拟测试和研究证据 | 不升级、不更改原证据、回滚机制保留 |
| 使用独立工业设备数据证明某个自动算法优于固定基线 | **算法/设备独立资格**，不是框架能否安装和稳定运行的前置条件；仍需真实 Paderborn 类多设备数据 |
| 生产工业设备闭环控制/航天飞控/安全关键自动判决 | 不支持，无认证，不纳入软件 Stable 承诺 |

## 3. RC0 必过门禁（FAIL/CANNOT VERIFY 就不发布正式 Stable）

1. **版本一致**：`source_product/pyproject.toml`、`phm_design.__version__`、安装后 `importlib.metadata` 都为 `0.8.0rc1`；README 明确 PRE-RELEASE，不能把版本候选当正式 Stable。
2. **发布介质**：在独立 build 环境成功构建 wheel 和 sdist；验证产物必需模块和元数据；计算 SHA256SUMS，仅作为 CI artifact 暂存，不默默将候选上传 PyPI 或公开 GitHub Release。
3. **真正的干净安装**：Linux Python 3.11/3.12 + Windows Python 3.11 通过从构建出的 `wheel` 安装（不是 editable/PYTHONPATH 下偷用工作区源代码）；`pip check` 必须通过。
4. **真正的算法提供者**：从独立公开仓库 `clonetimer/torch_like` 锁定 commit `580133dddf61cce7c354c648d4885e6caec99418`，运行 4 种原生候选（LogisticRegression、DecisionTreeClassifier、RandomForestClassifier、IsolationForest）的 fit/predict，并在可信本地模型制品上进行保存、SHA 和重载预测一致性验证。跨仓库任何网络或安装失败必须可见，不能默认为 skipped/pass。
5. **闭环操作性**：在源代码工作目录之外，以已安装 wheel 的模块入口执行 `--help`、TRAIN-only `design_cli propose` → 冻结 study SHA → `source-study` 真实提供者拟合 → 结果 JSON；错误使用完整保护角色数据必须拒绝且不创建输出。
6. **既有功能不回退**：运行 `tests/phm_design` 完整旧版+新代码 Python 组合回归，失败不能忽略；区分确因可选 Basilisk/工业数据/torch_like 缺失而跳过的测试，并在上线声明中写出跳过范围。
7. **版本控制与发布清晰**：同一提交构建 wheel/sdist，正式 RC 发布需要在 PR/发布记录中明确代码 SHA、制品 SHA、依赖、运行平台、已知限制、批准/拒绝和回滚地址。

此次 CI： [`.github/workflows/stable-candidate-080.yml`](../../.github/workflows/stable-candidate-080.yml)；干净 wheel 真正跨仓库 smoke 程序： [`scripts/stable_candidate_smoke.py`](../scripts/stable_candidate_smoke.py)。

## 4. 用户实际操作（候选环境，无额外编程）

**仅用于研究软件验收，不进行工业设备控制。** 安装前使用 Python 3.11（Windows/Linux），先按 CI 固定依赖安装可信 CPU PyTorch 和独立 torch_like，然后在 GitHub Actions 暂存制品中获取已构建 wheel。

```powershell
# Windows PowerShell (在独立新目录执行)
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install numpy==2.3.5 scikit-learn==1.8.0 scipy pandas PyYAML joblib
# 然后按上文钉住的 torch_like commit 安装经审核的提供者
.\.venv\Scripts\python.exe -m pip install --no-deps .\torch_like
# 将 CI artifact 解压得到发行 wheel 后执行（名称按实际构建产物）
.\.venv\Scripts\python.exe -m pip install --no-deps .\phm_design-0.8.0rc1-py3-none-any.whl
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m phm_design --help
.\.venv\Scripts\python.exe -m phm_design.industrial_contracts.design_cli --help
```

如只使用现有 0.7.0 已合格受限离线仿真，不要求安装实验性 torch_like；用既有 0.7.0 已冻结发布包。若希望第三方算法在 AAS 主程序自动注册，仍需另走 AAS 安装/兼容性验收流程。

## 5. 当前 Release Go/No-Go

`0.8.0rc1` **目前为发布候选**，不能因提交代码、合并 main 或上一轮 848 个测试通过就直接称为 Stable。只有以上**实际** Actions 验证通过，才可考虑打 RC tag、编制发布报告，再计划 `0.8.0` 正式软件框架 Stable。RC 并不保证模型性能提升；它保证的是如实列明范围的软件一致性和安装运行性。

候选期间：核心接口冻结，允许**修复测试阻断和文档/安装/打包问题**；禁止借发布收敛引入 R100 新模型、扩大工业设备资格或随意更改已冻结遥测审计签名。
