# 完整实验资料与Git LFS

本目录与data目录已按用户要求同步Windows中现有的完整相机图片、数据集、实验记录、原始浮点张量、日志和压缩备份。大型与二进制文件使用Git LFS，JSON/CSV等小型元数据保留为普通Git文件。

```bash
git lfs install
git lfs pull

# 也可只下载一类资料
git lfs pull --include="data/**"
git lfs pull --include="results/**"
```

仅运行实时算法时，可用GIT_LFS_SKIP_SMUDGE=1克隆；默认部署模型和程序仍为普通Git文件。使用仓库中的验证图片前，需要先拉取对应LFS对象。

文件清单和SHA256见[完整资料清单](../docs/full-data-manifest.json)。本次以Windows已保存资料为准，开发板192.168.2.221在同步时SSH超时，未核对或补充板端独有文件。

npu-bottleneck-v1是第三轮NPU定位及超分优化；optimization-v2是第二轮优化。qualification-20261124/job.json保留原完整验收的意外重启中断状态，上传原始资料不会将该状态改为通过。
