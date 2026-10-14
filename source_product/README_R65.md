# PHM AutoDesign R65 — Thermal 原生热激励开发实验 / 0.8.0.dev6

**2026-10-13；状态：DEVELOPMENT ONLY。** 以R64的`0.8.0.dev5`源码为基线；不修改冻结的0.7 Stable。不发布RC，不新增诊断资格，不开展新的独立封存评价。本轮选取的是原《R65_PRE_REGISTER_DRAFT.md》明确允许的**先导开发试验**，并非计划中8/8/16世界训练/校准/封存阶段。

## 1. 实际软件改动

- `src/phm_design/thermal_excitation_r65.py`: 定义预设热功率命令A(旧两档)和B(八档、220秒段)。它只计算**已知加热命令**，不模拟温度；实际热状态由Basilisk2.11.0 `SensorThermal`执行。
- `src/phm_design/adapters/sat_multicomponent.py`: `run_native(...,thermal_excitation=None)`新增显式可选热激励参数；默认旧模式字节级数值保持不变，非Thermal域拒绝指定激励，全部故障仍调用原Basilisk热部件。
- `research/R65_DEV_PROTOCOL.json`: 开发试验协议，**事先提交git**再运行新世界，方案SHA256见下方证据。
- `research/run_r65_dev_experiment.py`: 4个新开发物理世界×2激励臂×5类标签，各运行同噪声物理配对和独立噪声观测，共80条原生轨迹。原始研究oracle与部署字段结构分别归档，没有训练/选模。
- `research/verify_r65_dev_evidence.py`: 校验全部83项受验文件、80个CSV的字段/181采样行、元数据/开发身份，拒绝缺损文件和额外文件。
- `research/explain_r65_thermal_timescales.py`: **后验解释性分析**，辐射局部线性化时间常数，与预注册评价指标分离，不会修改已有实验结果。
- `tests/phm_design/test_r65_thermal_excitation.py`和`test_r65_timescales.py`: 旧模式精确兼容、激励按时间规定、篡改合同拒绝、非热域拒绝、Basilisk轨迹变化、研究身份检查。0.8.0.dev6 wheel/sdist已构建并经定向安装检查。

## 2. 前置冻结与实际执行

- 冻结实验Git提交：`eeb130452453b11f5476ebc83c54cd873461119b`。
- 预先冻结协议SHA256：`88dfae727faf563386bb9d510d10c4e7184203ae8426a5cea2521a9b96198075`。
- 物理世界ID：`r65-thermal-dev-pilot-001`到`004`；同一世界的两臂物理参数相同，Basilisk版本固定2.11.0。样本频率20s、每条轨迹181点、仿真3600s；注入与真值仅供研究审计。
- 每臂每世界5条同噪声数值对照，另5条按`world:arm:label`派生的独立噪声观测，共**80条**。
- 旧R62已经揭盲的四世界**未参与此次运行、选模或评分**。R65没有创建校准/盲测世界，模型训练次数0。
- 原始CSV包含六个字段：`time_s, illumination, heat_a_command_w, heat_b_command_w, temperature_a_k, temperature_b_k`；不包含仿真故障标签、真实故障起始、注入幅度或反事实真值。

## 3. R65预设物理指标（4开发世界×2部件，每臂8个机理对照）

|中位数指标|A旧两档|B八档/220s|
|---|---:|---:|
|健康段加热档位|2|7.5（范围7—8）|
|健康段温度跨度K|15.9125|15.2417|
|故障后正常轨迹温度跨度K|18.3238|17.8678|
|发射率下降/额外热输入配对响应Pearson|0.9992630|0.9992732|
|最佳缩放响应残差比例|0.030218|0.030147|
|局部常量/T⁴设计矩阵条件数|40.10|39.98|

- **结论：** 指令档位增加，但温度跨度和机理响应形状没有明显改善。不能声称增强激励诊断性能提升，也不能从高相关推导结构不可辨识的数学定理。
- 每条轨迹包含模拟`0.1K`高斯温度观测噪声；用于物理差分的配对组同随机数流噪声抵消，因此其差分指标不属于实测检测精度。独立噪声CSV已保存，但本轮**没有分类器的泛化风险估计**。
- **后验物理解释（不是冻结指标）：** 按辐射项局部线性化估算的热特征时间为2447—7384s，中位4652s，远长于220s激励段。可能存在动态带宽不匹配；需要后续另外冻结的验证实验。
- A/B部件为未动态耦合的低保真模型，不能外推真实整星热网络。

## 4. 测试与发行状态

- Python3.13.5、已实际安装Basilisk2.11.0环境，受限BLAS线程的本轮全套回归：**776 PASS/0 FAIL/0 SKIP**（含时间常数解释性分析测试）。
- wheel/sdist均成功构建，已以`--target`安装wheel检查：`0.8.0.dev6`、`DEVELOPMENT`；捆绑0.7模型在新包中显示`qualified=false`。
- `pip check`**未通过**：缺sgp4，Basilisk对当前Pillow版本不兼容，宿主moviepy也不兼容。已上传的Pillow wheel是PyPy ABI、不能给CPython使用；`sgp4-2.25.whl`文件名不符合wheel正式命名规范。本轮不伪称完整依赖闭包已验收。
- Python3.11/3.12×wheel/sdist完整发行矩阵未复跑，不允许升RC/Stable。
- 本次仅使用本地Git feature分支进行研发，合并/tag记录随包提供；**未推送远程GitHub仓库**。

## 5. 恢复与运行

在Basilisk2.11.0以及PyTorch、NumPy、SciPy、sklearn等依赖可用的隔离环境，进入`source_product/`：

```bash
PYTHONPATH=src python research/verify_r65_dev_evidence.py --root ../evidence/r65_native_dev
PYTHONPATH=src python research/explain_r65_thermal_timescales.py \
  --simulator-parameters ../evidence/r65_native_dev/SIMULATOR_PARAMETERS_ONLY.json \
  --output /new/dir/R65_POSTHOC_TIMESCALE.json
# 若要复做开发试验，必须选不存在的新输出目录。
PYTHONPATH=src python research/run_r65_dev_experiment.py --output /new/path/r65_dev_reproduction
# 避免意外重开任何封存世界；新试验仍仅是开发试验。
```

`evidence/r65_native_dev/` 存储全部原始轨迹、研究参数、指标和文件哈希；`evidence/` 存放测试与打包日志。`research/R66_NEXT_EXPERIMENT_DRAFT.md`提供下一步物理时标匹配假说；**不是已冻结的R66实验协议**。
