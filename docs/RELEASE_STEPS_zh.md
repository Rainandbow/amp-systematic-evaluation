# 上传、版本与DOI

1. 先将当前目录内的文件和子目录上传到现有 GitHub 仓库。浏览器上传前确认没有选中 `analysis/inputs/`、本地运行结果或整个论文资料文件夹。
2. 在 README 和 DATA_ACCESS 中补充公开输入数据的获取方式；确认原创代码许可和实际软件作者。版本号不能代替数据获取地址。
3. 登录 Zenodo 并关联 GitHub；进入个人菜单的 GitHub 页面，点击 Sync now，开启本仓库的归档开关。
4. 回到 GitHub 的 Releases，创建新标签 `v1.0.0`，Target 选择完成核查的代码提交，填写发布说明并发布。
5. Zenodo 处理完成后打开对应 v1.0.0 的记录，核对作者、版本、许可和实际归档文件，复制该版本 DOI。
6. 将 GitHub 仓库链接、v1.0.0 和真实版本 DOI 填入论文。此处不预填虚构 DOI。

GitHub仓库：https://github.com/Rainandbow/amp-systematic-evaluation

官方说明：
- https://help.zenodo.org/docs/github/enable-repository/
- https://help.zenodo.org/docs/github/archive-software/github-upload/
- https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
