# 抗菌肽预测模型系统评价：代码上传准备版

对应仓库：https://github.com/Rainandbow/amp-systematic-evaluation

本目录尚未发布 GitHub Release，也没有 Zenodo DOI。所有代码和统计表由已有归档整理，不重新计算论文结果。

## 已包含

- `analysis/`：六方向序列基线、家族内 AUROC、分组隔离筛选验证代码与固定配置。
- `figures/`：从已完成的正式统计表重建主图4、主图5及补充图S20–S24的代码、统计表和图形检查工具。
- `historical_reference/`：历史比较、预算敏感性、阈值任务敏感性和旧版图件的参考代码。
- `tools/import_local_inputs.py`：核对 SHA256 后导入已有本地分析包中的八份输入。

## 本地核查与绘图

```powershell
python -m pip install -r requirements.txt
python run.py check
python run.py test
python run.py figures
```

前两项核查不需要 GPU 或模型权重。绘图使用已保存的正式结果，不重新推理或 bootstrap。图件输出在新建的 `figures/output_*` 目录。

## 完整复算的数据状态

原始序列、逐条实验标签和原始来源文件没有放入这个上传目录。已有的完整内部分析包仍保留在本机，八份输入的文件名和哈希记录在 `analysis/input_manifest.json`。

导入现有本地包后可以执行完整新增分析：

```powershell
python tools/import_local_inputs.py --source "已有amp_baseline_honest分析包.zip的完整路径"
python run.py analysis-check
python run.py baseline
python run.py transfer
```

这八份输入导入到 `analysis/inputs/` 后由 `.gitignore` 排除。需要先确认第三方数据的再分发条件，再为数据补充公开获取地址；当前不能将本仓库描述为已经包含全论文的全部可复算输入。

## 上传方式

在 GitHub 点击 `Add file → Upload files`，拖入本目录内的文件和子目录，确认顶层能直接看到 `README.md`、`run.py`、`analysis/` 和 `figures/`，然后提交。本目录含多个子目录，若浏览器提示文件数量上限，分批上传即可。

完成代码和数据获取说明后，再连接 Zenodo 并发布正式版本。准确操作见 `docs/RELEASE_STEPS_zh.md`。
