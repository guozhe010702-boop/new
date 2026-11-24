# 已公开的测试记录

仓库保留经过筛选的JSON、CSV、逐层性能表和恢复记录，用于核对报告数值。完整相机图片、预览、原始浮点张量、内核大日志与压缩备份保留在原Windows/开发板目录，没有提交到Git。

`npu-bottleneck-v1`为第三轮NPU定位及超分图优化，`optimization-v2`为第二轮CPU与显示优化。`qualification-20261124/job.json`保留原完整验收的意外重启中断状态；本仓库不声明三项目已完成长时间稳定性验收。

`data/dataset.json`只保存校准/验证图片的来源说明与散列，验证集图片本身没有公开。独立性能分析可通过`tools/npu_benchmark.py --image`传入自己准备的对应模型尺寸图片；重做完整50图对比需要原验证集。
