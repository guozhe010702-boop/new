# RK3588 三项视觉算法

GitHub仓库：[guozhe010702-boop/new](https://github.com/guozhe010702-boop/new)。包含源码、默认RKNN模型、原超分回退模型、已测试ARM64运行程序、私有运行库、文档和筛选后的测试记录。相机图片、数据集、原始浮点输出与大体积日志保留在本地。当前版本经过短时复测，长时间稳定性验收尚未完成。

在IMX415相机和RK3588上独立运行单目深度、轻量超分、YOLO11与ByteTrack。C++负责采集、RGA预处理、RKNN推理、后处理及HDMI显示；Python负责监督、遥测、模型转换和验收。三个项目轮流运行，默认不开机自启。

## NPU第三轮优化

实时超分默认使用相同权重的FP16卷积图与CPU尾部，单上下文180秒复测稳态约11.81FPS；以`--sr-tail npu`可使用原完整模型。50张验证图与原FP16模型输出一致。深度已定位到注意力单核热点，当前仍约1.42FPS。代码、原始数据、切换命令和限制见[NPU瓶颈定位与第三轮优化](docs/NPU瓶颈定位与第三轮优化.md)。长时间稳定性验收尚未完成，后台预览保持停止。

## 第二轮优化状态

代码和板端二进制已更新；超分向量化后处理与显示缓存复用已完成交替对比。当前采用单上下文保守配置；原完整验收因开发板意外重启中断，三项目当前版本的长时间稳定性仍未完成。实测与限制见[第二轮优化报告](docs/第二轮优化报告.md)。后台预览与验收均已停止，不自动重启。

## 程序位置与入口

Windows源码目录：`C:\Users\dvsense_docker\Desktop\work\RK3588\av_dev\advanced algorithms`。

开发板目录：`/userdata/av_dev/advanced algorithms`。路径有空格，终端中请保留双引号。

```bash
cd "/userdata/av_dev/advanced algorithms"
# 三选一，启动前先停止上一个算法
bash yolo11_bytetrack/start.sh
bash real_esrgan/start.sh
bash depth_anything_v2/start.sh

# 三个项目共用停止和状态入口
bash yolo11_bytetrack/status.sh
bash yolo11_bytetrack/stop.sh
```

启动创建临时`advanced-algorithms.service`，关闭SSH窗口不会停止；停止脚本会终止程序并恢复原桌面、显示布局和相机格式。请使用root。运行期间暂时独占HDMI，原桌面服务由监督器恢复；不要直接强杀C++进程作为日常停止方式。

正式验收期间由`advanced-qualification.service`占用设备，其他启动请求会被拒绝。`stop.sh`也可以中止验收，中止会记录为interrupted，不会标为通过。

## 模型与默认精度

| 项目 | 模型输入 | 输出 | 精度选择依据 |
|---|---|---|---|
| YOLO11与ByteTrack | 640×640 RGB letterbox | COCO检测、轨迹ID、双向越线事件 | INT8通过小子集量化精度对比 |
| Real-ESRGAN | 256×256 RGB ROI | 1024×1024 RGB | FP16；本次INT8未通过SSIM门槛 |
| Depth Anything V2 Small | 518×518 RGB letterbox | 稠密相对深度 | FP16；本次INT8未通过秩相关门槛 |

深度值不带米制尺度；不据此输出未经标定的障碍物距离。超分是学习型图像增强，不证明生成细节是真实物理结构。

默认相机输入为1080p，GStreamer通过`videorate drop-only`限制到30帧采样，不把模型输入叫作1080p推理。原始传感器可能仍工作在60Hz；实际处理FPS以统计为准。当前HDMI屏幕为1080p60，不能据此声称4K60显示。

## 操作示例

前台运行60秒，结果目录必须是新目录：

```bash
python3 tools/control.py run --project yolo11_bytetrack --seconds 60 --output "results/my-yolo-test"
```

图片、视频文件和配置示例：

```bash
python3 tools/control.py run --project depth_anything_v2 --image /userdata/test.jpg --hdmi 0 --seconds 5
python3 tools/control.py run --project yolo11_bytetrack --video /userdata/test.mp4 --hdmi 0 --seconds 60
python3 tools/control.py start --project yolo11_bytetrack --class 0 --line 0.5
python3 tools/control.py start --project real_esrgan --roi-x 100 --roi-y 100
```

`--class 0`为人，`--class -1`跟踪所有COCO类别；不同类别分开关联。计数线为归一化纵坐标，默认0.5，框底边中心跨越滞回区后分别累计两个方向。首版计数线为水平线。

`--workers 1 --mask 1`为单核，`--workers 1 --mask 7`为一个上下文允许使用三核，`--workers 3`为三个上下文分别绑核。允许三核不代表每个网络都会自动获得三倍性能。各项目`recommended.json`由实际调度对比和验收生成；没有推荐文件时使用FP16、一个上下文、mask7。

## 离线整图超分

```bash
LD_LIBRARY_PATH="$PWD/runtime/lib" ./build/sr_tiled \
  real_esrgan/models/fp16.rknn /userdata/input-4k.png /userdata/output-x4.tif
```

输入3840×2160时输出15360×8640。256输入块、40像素上下文、176像素有效步长，边缘块移到真实图像边界后裁剪。最大支持4096×4096输入，任一边至少256像素。输出像素数组在内存中组装：4K输入的×4输出仅RGB数组就需约398MB，保存时还有额外内存。本功能不承诺4K整帧实时。

## 构建与模型转换

板端系统依赖首次缺失时执行`bash tools/install_system_deps.sh`。停止算法后执行：

```bash
bash tools/build_board.sh
```

编译链接工程私有`runtime/lib/librknnrt.so`和`librga.so`，不覆盖系统旧版本。构建前拒绝替换正在运行的本工程二进制。OpenCV、GStreamer及libdrm由板端系统提供，BSP相机、ISP/IQ、RKAIQ和NPU驱动必须已经可用。

转换在Windows主机已有的x86_64 Docker镜像`uav-yolo11-converter:2.3.2`中进行。PowerShell在工程根目录执行：

```powershell
docker run --rm --entrypoint python3 --mount "type=bind,source=$PWD,target=/project" uav-yolo11-converter:2.3.2 tools/convert.py --project yolo11_bytetrack --precision fp16
```

将项目名替换为`real_esrgan`或`depth_anything_v2`可转换其他模型。INT8前先运行`tools/prepare_data.py`准备代表性数据，再加`--precision int8`；不使用随机图片替代校准集。原始官方权重可通过`tools/fetch_models.py`按SHA256核验恢复。重建转换镜像可参考同级原工程`UAV Object Detection/conversion`的锁定Dockerfile和依赖。

## 结果与报告

每次运行在`results`下产生独立目录：

- `frames.csv`：预处理、推理、后处理、显示耗时及取到GStreamer样本后的帧龄。
- `tracks.csv`、`events.csv`：跟踪框、ID、类别及越线事件。
- `summary.json`、`analysis.json`：累计与分位数、资源及恢复分析。
- `telemetry.jsonl`：原生进程CPU时间、RSS、FD、线程、温度及NPU负载。
- `input-*`、`prepared-*`、`preview-*`：选定帧的原图、模型输入和预览。
- `depth-*.yml.gz`：OpenCV浮点相对深度数组；`sr-*.png`为超分图。
- `restoration.json`：设备和桌面恢复结果。

内部帧龄起点是appsink取样，不包含曝光、传感器读取和取样前积压；首帧设置模式用应用时刻，后续HDMI帧以真实翻页事件计时。不宣称光学端到端延迟。

验收状态和最终数据见`docs/实测报告.md`。处理流程、内存边界及文件作用见`docs/处理流程与接口.md`。模型、依赖及修改后的第三方源码散列在`docs`中保留；YOLO、ByteTrack、Depth Anything及Real-ESRGAN各自的许可证见`vendor`。
