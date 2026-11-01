# 当前六张主图与修订诊断

本次v1.1.0候选版在原有仓库中新增完整目录six_main_figures_v1_1_20261101，原有代码和输入保留。

当前论文六图入口：先进入该目录，再执行python run.py check、python run.py points、python run.py figures。根目录原有run.py仍对应v1.0.1的基线/家族、分组验证和旧七图子集，不能混用。具体依赖与逐图登记见新增目录README及docs/MAIN_FIGURE_REGISTRY.tsv。

revision_diagnostics保留表S11绝对验证、表S12最短优先48组配对差、表S13/S14与图S27家族和块删除诊断。公开版第三步历史输入已替换为SHA256标识，29,104行其余单元格和顺序未变，14张删除分析科学结果重建逐字节一致。原图4/5统计另有14张正式表复算一致；两组核验不是同一组表。

原始v1.0.1 DOI为https://doi.org/10.5281/zenodo.23836777。v1.1.0需发布后核对实际Zenodo记录并取得版本DOI，不能沿用旧DOI声称旧包包含新增内容。

代码Apache2.0、指定处理后统计数据CC BY4.0及第三方资料各自权利的范围，见现有NOTICE、docs/LICENSING.md和docs/SOURCES.md。历史原序列、原始工作簿和模型权重按原始来源获取。公开不等同于能从SHA256恢复序列或重新推理。
