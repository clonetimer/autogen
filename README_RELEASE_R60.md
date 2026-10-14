# PHM AutoDesign 0.7.0 Stable — 使用与恢复指南

## 开始之前

这是 **SAT/Basilisk 仿真域有限支持稳定版**，支持 Linux x86_64 上的 CPython 3.11 与 3.12。诊断要求整段轨迹和可核实的初始健康前缀。**不要用于真实航天器飞行安全决策或在线实时闭环控制。**

交付包包含当前0.7.0 wheel/sdist、历史开发候选、原0.6.0回滚wheel、完整产品源代码、校准/封存世界原始数据与评分、RC/Stable四格安装回归、100项操作、失败记录和验证器。没有把外部依赖的大型Torch/Basilisk wheelhouse重打包；可复用此前已经验证的 CPython 3.11/3.12 离线依赖。

## 1. 检查交付包

```bash
python VERIFY_R60_RELEASE.py
sha256sum -c MANIFEST_SHA256SUMS.txt
```

验证只读取文件并检查字节、证书、物理世界独立性、模型哈希、实际测试与回滚日志，不运行仿真、不重训、不打开新的封存世界。

## 2. 安装 Stable

在已有依赖或可访问合规仓库的独立环境中：

```bash
python3.11 -m venv .venv-phm-070
. .venv-phm-070/bin/activate
python -m pip install "artifacts/phm_design-0.7.0-py3-none-any.whl[dev,simulation,automl]"
python -m pip check
phm-design-limited scope
```

完全离线安装时使用早前补齐并验证的对应ABI wheelhouse，例如：

```bash
python -m pip install --no-index --find-links=/absolute/path/wheelhouse \
  "artifacts/phm_design-0.7.0-py3-none-any.whl[dev,simulation,automl]"
```

sdist同样支持，可将安装文件替换为 `artifacts/phm_design-0.7.0.tar.gz`。不要使用 PyPy Pillow wheel 假冒 CPython ABI。

## 3. 查询支持范围

```bash
phm-design-limited scope
phm-design-limited check-use-case --case ADCS:cross_scenario
phm-design-limited check-use-case --case Thermal:weak_fault
```

第二个命令应标记仿真域受支持，第三个返回 `OUT_OF_CLAIM_NOT_QUALIFIED`。未认证的项目不能强制宣布诊断合格。

## 4. 用真实符合合同的 CSV 推理

```bash
phm-design-limited predict-bundled \
  --domain EPS \
  --case EPS:progressive_degradation \
  --observations /absolute/path/to/eps_deployment_observations.csv \
  --healthy-prefix-confirmed
```

CSV 必须恰好包含目标域部署观测白名单和有序列；字段名称可从 `phm-design-platform inspect` 获取。不要把训练标签、故障发生位置、注入程度、仿真真值、反事实健康轨迹送入在线预测字段。只有确有依据确认初始20%健康时才加入 `--healthy-prefix-confirmed`；不满足该条件时应保留UNKNOWN/不提供已认证预测。

**注意：**认证支持的是在相关仿真分布中整段轨迹的离线判别，不能据此声称实时预警或在线逐采样因果分类。

## 5. 训练一个新模型与资格标签区别

产品仍提供通用算法设计入口，例如：

```bash
phm-design-limited design \
  --dataset /absolute/path/to/your_design_train.json \
  --output /absolute/path/to/new-empty-output-directory \
  --trials 24
```

此命令生成的新模型仍为 **DEVELOPMENT_ONLY**，它不会自动继承捆绑五个模型的 15项诊断资格。训练数据合同需包含世界分组、离线标签及 `design_train` 角色；`--trials`只是候选个数，不是硬CPU预算。

## 6. 回滚

使用交付包内字节完整的原0.6.0 wheel，在独立或维护窗口执行：

```bash
python -m pip uninstall -y phm-design
python -m pip install --no-index --no-deps \
  artifacts/phm_design-0.6.0-py3-none-any.whl
python -m pip check
python -c 'import phm_design; print(phm_design.__version__)'
```

预期版本 `0.6.0`。若当前运行环境的依赖集合与旧版本合同不同，还需要按照旧0.6.0的环境资格要求恢复兼容依赖。实际跨版本回滚验证记录在 `evidence/312_sdist_ROLLBACK_*`。

## 7. 接续研究与模型支持范围

现有发行仍不支持ADCS/EPS弱故障和全部Thermal诊断资格。旧R52–R59研究包应单独保存，不能将过去失败评价世界重新命名为全新盲测。新的研究模型与此 Stable 的资格是两个不同对象；在独立训练/校准/封存重新取得证据前，不能扩张Stable支持范围。

参看 `RELEASE_DECISION_R60.md`、`FINAL_RELEASE_AUDIT.json`、`QUALIFICATION_CERTIFICATE.json` 和 `R60_15_CELL_QUALIFICATION.csv`。除非需要复核物理模型参数，否则日常安装和推理不需要重新运行900个资格世界。
