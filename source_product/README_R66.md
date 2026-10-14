# PHM AutoDesign — R66 研发交付与实证结论（0.8.0.dev7）

**状态：DEVELOPMENT_ONLY，物理激励假说未通过预设探索性门槛。** 0.7.0 SAT/Basilisk 仿真域有限支持 Stable 未修改；R66未训练模型、未开启校准或封存盲测世界、没有新增诊断资格。

## 1. 目标与假说

R65的220s多级加热虽然增加了指令档位，仍没有提升故障机理分离。R66在新开发世界中比较同样总时长的原始双档加热 (A) 与2400s长周期加热 (B)，以判断增加热状态覆盖能否改善同部件发射率损失 / 额外热输入的物理响应差异。

**实验控制：** 两组总时长都为14400s、采样间隔20s、积分步长2s、故障激活时刻7200s、相同世界物理参数、相同太阳边界条件和故障强度。B组各档倍率预定为[0.50,1.70,0.55,1.80,0.60,1.65]，B部件指令相位偏移3档；A组原始双档加热模式不改变。使用四个全新开发世界 `r66-thermal-timescale-dev-001` 到 `004`；它们不属于R65或历史揭盲的R62世界。

冻结的实验协议：`research/R66_DEV_PROTOCOL.json`，SHA256 `5474315bc5f2c34dcf90da05f6cb271decdffee4db95130a4aa6b9340c02d3da`。先冻结协议和实现代码后执行仿真。所有原始轨迹来自实际安装的 Basilisk 2.11.0 `SensorThermal`，新增模块只计算预设加热命令，不以纯Python公式替代温度积分。

## 2. 原生轨迹、数据边界与完整性

4个世界×2组×5类轨迹×2种噪声构造 = **80条Basilisk轨迹**（每条721行、6字段）。

- `research_oracle_pairs`：40条同种子观测噪声配对，用于计算相对健康轨迹的物理差分；此类配对**不允许**进入部署推理或模型训练。
- `deployment_observations_dev_only`：40条独立噪声观测，公开字段仅为`time_s, illumination, heat_a_command_w, heat_b_command_w, temperature_a_k, temperature_b_k`。本轮**没有**使用其训练模型或获得诊断精度。
- `SIMULATOR_PARAMETERS_ONLY.json`：真值故障与物理参数仅供研究审计；注入时刻、故障标签、强度、反事实健康轨迹不得加入推理字段。
- `MANIFEST_SHA256.json`：83个受验文件；`verify_r66_dev_evidence.py`核查数量、哈希、原始轨迹、预注册协议、禁止误提升诊断资格。

## 3. 预设评价指标与真实结果

| 指标（部件配对的中位数） | A 双档 | B 长周期 |
|---|---:|---:|
| 故障前温度跨度 K | 37.5970 | 44.5379 |
| 同部件机理响应 Pearson | 0.993619 | 0.993384 |
| 最佳标量拟合后机理响应残差 | 0.056222 | 0.057759 |
| 常量 / T⁴ 条件数 | 56.4894 | 33.6929 |

- 预设**主指标**是8组配对改善量的中位数（B残差−A残差），实际为 **+0.00126364**，未达到+0.02绝对差值门槛。
- 8个部件配对中只有**4/8**的残差改善，未达到预设至少6/8。
- 故障前温度跨度的**配对改善中位数为+5.823 K**，满足预设+5K辅助门槛，但无法代替主指标。
- 逐部件看：A部件4/4个世界残差变化为负，B部件4/4为正；B的结果不能被总体均值掩盖，后续需排除加热相位与部件物理参数的混杂。
- 根据既定“同时满足主指标、改善计数和温度跨度”的开发判断准则，`exploratory_physics_thresholds_met=false`。

**解释限制：** 所有机理指标来自研究oracle中同噪声反事实响应，故障发生时刻已知但仅用于离线审计。这不是诊断准确率、真实传感器指标或在轨可靠性证据；四个开发世界不足以构成高置信度总体效应声明。A/B改变了加热序列、幅值分布和切换周期，不能把观察到的差异单独归因为“时间尺度”一个因素；R67应采用能控制这些混杂因素的实验设计。

## 4. 实现、兼容性及测试

- `src/phm_design/thermal_excitation_r66.py` 新增长周期**命令**配置。
- `src/phm_design/adapters/sat_multicomponent.py` 仅新增明确的 `thermal_duration_s` 参数（3600—28800s、20s整数倍、仅热域）与R66命令分支。**默认**仍按历史3600s路径执行，不改变旧模型算法、标签或0.7资格。
- `research/run_r66_dev_experiment.py` 在全新目录中运行开发实验；不覆盖已有输出。`research/verify_r66_dev_evidence.py`独立验档。
- 代码全量Python3.13.5/Basilisk2.11.0已安装环境：**786 PASS、0 FAIL、15条非致命警告**；包含10项新的R66定向测试。
- wheel、sdist构建成功；wheel目标目录安装后显示版本0.8.0.dev7及`deployment_qualification_status=DEVELOPMENT_NOT_QUALIFIED`，从sdist也可再次构建wheel。
- 宿主环境 `pip check` **失败**：缺`sgp4`，当前 Pillow 12.3.0 不符合bsk 2.11.0和moviepy 2.2.1的约束。Python3.11/3.12四格发行环境尚未验证；禁止升RC。

## 5. 使用

```bash
cd source_product
PYTHONPATH=src python research/verify_r66_dev_evidence.py --root ../evidence/r66_native_dev
# 仅需要在新的、空路径重新进行开发仿真时才运行；不能当作盲测
PYTHONPATH=src python research/run_r66_dev_experiment.py --output /new/development/output
```

## 6. 阶段判断与下一步

**R66工程闭环完成，物理假说没有通过开发门槛；0.8.0.dev7继续 DEVELOPMENT_ONLY。** 不建议因为条件数下降就直接打开资格盲测、扩充分类器或宣称故障可区分。R67应针对每个部件交叉交换加热相位，并尽可能控制平均加热能量与持续时间，先判断本次A/B方向相反的原因，再决定是否开展新的诊断模型训练。
