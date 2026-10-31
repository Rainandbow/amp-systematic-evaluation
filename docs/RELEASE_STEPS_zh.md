# 发布本次许可与输入更新

1. 提交这份完整目录到现有仓库，保留已有 v1.0.0 标签及归档。
   顶层应直接看到 LICENSE、NOTICE、README、analysis、figures 和 docs。
   `analysis/inputs/` 本次是经过核查的公开处理输入，需一并提交。
2. 确认本次提交后的 `python run.py check`、`analysis-check` 和 `test` 通过。
3. 在 GitHub Releases 创建新标签 `v1.0.1`，Target 选择本次完成核查的提交。
   标题建议：v1.0.1 — licensed code and processed statistical inputs。
4. 发布说明写明：软件 Apache-2.0；处理数据 CC BY 4.0；新增八份输入；
   两份历史表使用 SHA256 序列身份；统计方法、种子和冻结结果不变。
5. Zenodo 的 GitHub 归档开关已启用。归档完成后核对新版本的文件、作者
   Zhiyu Jin / Feifei Xiao、软件许可 Apache-2.0，以及描述中的数据许可例外。
   `.zenodo.json` 已提供相应元数据；不预填新版本 DOI。
6. 将实际生成的新版本 DOI 填入 README 和论文 Code/Data availability。
   不能将 v1.0.0 DOI 写成含有新增八份输入的归档。

软件许可默认值对应代码；CC BY 4.0 只对应指定处理数据和其注明的来源。
不要将第三方模型、权重或原始数据集合宣称为本项目 Apache 2.0 软件。

已有归档：https://doi.org/10.5281/zenodo.23834121
创建新版本：https://github.com/Rainandbow/amp-systematic-evaluation/releases/new
