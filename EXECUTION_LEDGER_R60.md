# R60 稳定版最终执行台账

| 阶段 | 已完成工作 | 记录和结论 |
|---|---|---|
| 范围授权 | 用户同意15项最终支持范围，旧16项失败合同封存 | `APPROVED_15_CELL_CONTRACT.json`；原R59失败记录保留 |
| 冻结与校准 | dev13与五个模型冻结；新300世界980轨迹，15/15通过 | `qualification/PREQUALIFICATION_FREEZE.json`、`CALIBRATION_*` |
| 一次性封存 | 校准通过才解锁；新600世界1960轨迹，15/15通过 | `qualification/SEALED_FREEZE.json`、`SEALED_*` |
| RC1 | 构建RC wheel/sdist；3.11/3.12四格738 PASS，无失败无跳过 | `artifacts/*rc1*`、`rc_evidence/` |
| Stable构建 | 0.7.0 wheel/sdist；模型哈希不变，仅允许发行元数据改变 | `STABLE_BUILD_RESULT.json`、制品差分验证 |
| 首次Stable操作失败 | 缺少合成Thermal_localization测试样例，首轮311_wheel 24/25；保留原错误 | `failed_attempts/311_wheel_first/`、`TEST_HARNESS_AMENDMENT_001.json` |
| Stable正式复验 | 3.11/3.12 × wheel/sdist：各738 PASS，25项稳定性操作PASS，Basilisk冒烟PASS，pip check PASS | `evidence/`、`STABLE_OPERATIONS_*.json`、`STABLE_MATRIX.log` |
| 回滚 | 在3.12 sdist环境从0.7.0实际卸载，安装原哈希0.6.0并检查依赖与版本 | `evidence/312_sdist_ROLLBACK_*` |
| 本轮最终只读审计 | 校准和封存15项；900世界ID/参数指纹独立；100项操作唯一、无超限；五模型权重冻结；两版release core差异受控 | `VERIFY_R60_RELEASE.py`、`FINAL_RELEASE_AUDIT.json` |
| 发行判定 | `PASS_SIMULATION_DOMAIN_STABLE` | `RELEASE_DECISION_R60.md` |

**计数说明：**四次738项回归来自四个环境，不代表2952个不同测试用例；100项稳定性操作按各环境25项去重累加；两组资格世界不与历史开发世界重复计作新训练。所有模型和评分均沿用当时冻结的算法，没有看到封存成绩后调参。此台账用于软件项目级发行复核，不构成在轨安全保证。
