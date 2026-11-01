# 抗菌肽预测模型系统评价：代码与统计复算输入

原创代码及归档的绘图代码统一使用 **Apache 2.0**；整理生成的数据使用
**CC BY 4.0**，保留原始文献署名和第三方权利。详见 `NOTICE`、
`docs/LICENSING.md` 和 `docs/SOURCES.md`。

本版已包含统计复算所需的八份文件，位于 `analysis/inputs/`。
六份保持冻结输入原字节；两份历史表将序列字母替换为同一序列的 SHA256，
其他单元格和行顺序保持不变。标签、分组、模型分数、预测和判定规则不变。

## 核查与复算

```bash
python -m pip install -r requirements.txt
python run.py check
python run.py analysis-check
python run.py test
python run.py baseline
python run.py transfer
python run.py figures
```

`baseline` 重算新增基线/家族分析及 10,000 次分组 bootstrap；`transfer`
重算六个留一来源折、每折 20 次分组隔离验证。新结果写入独立目录。
`figures` 单独使用已冻结正式结果重建主图4、主图5及补充图S20–S24，
不会自动替换为新复算的结果。以上命令不需要 GPU 或模型权重。

八份输入覆盖新增151候选基线/家族分析和历史分组筛选验证，不能据此声称
整篇论文所有图件均已从原始实验资料独立重建。历史原始序列重取、重新分组
和模型推理需要原始数据来源与第三方模型环境，见数据获取说明。

## 版本与引用

仓库：https://github.com/Rainandbow/amp-systematic-evaluation

已有 v1.0.0：https://doi.org/10.5281/zenodo.23834121

v1.0.0 是早期代码快照，不含本次新增输入。当前更新应通过新提交和新版本
归档；新版本 DOI 必须在 Zenodo 实际生成后填写，不能沿用旧 DOI 并声称
旧版本包含本次数据。发布操作见 `docs/RELEASE_STEPS_zh.md`。
