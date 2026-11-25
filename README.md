# RK3588 边缘视觉算法平台

基于 **RK3588 + IMX415** 的 C++ 视觉推理工程，集成 **Depth Anything V2 单目深度估计、Real-ESRGAN 图像超分辨率、YOLO11 + ByteTrack 多目标跟踪计数**。提供相机采集、RGA预处理、RKNN NPU推理、HDMI显示与手动运行管理。

仓库包含源码、默认RKNN模型、已测试的ARM64程序、项目私有运行库、完整本地数据与日志，以及可核对的性能报告和测试记录。当前定位为可运行的算法实验与部署工程，**长期稳定性验收仍未完成**。

[快速开始](#快速开始) · [实测性能](#实测性能) · [目录与核心程序](#目录与核心程序) · [详细文档](#详细文档)

## 三个算法项目

| 项目 | 模型与默认精度 | 模型输入 | 主要输出 |
|---|---|---|---|
| 单目稠密深度估计 | Depth Anything V2 Small，FP16 | 相机518×294；文件518×518 | 浮点相对深度、伪彩深度预览 |
| 图像超分辨率 | Real-ESRGAN general-x4v3，FP16 | 256×256 RGB ROI | 1024×1024 RGB，与双三次插值并排显示 |
| 多目标跟踪计数 | YOLO11n INT8 + ByteTrack | 640×640 RGB letterbox | 检测框、轨迹ID、双向越线计数及CSV事件 |

- **深度估计**：单目RGB生成稠密相对深度，支持保存浮点深度数据；输出未标定为米制距离。
- **超分辨率**：支持实时ROI增强和离线大图分块×4处理；离线4K输入可输出15360×8640图像。
- **目标跟踪**：默认跟踪人，支持选择COCO类别；使用框底边中心跨越水平计数线，累计两个方向的事件。

三个项目独立启动、轮流运行，共用相机、NPU和HDMI资源。默认不开机自启。

## 处理链路

```text
IMX415 / ISP → GStreamer采集NV12 → RGB转换 / RGA预处理
            → 最新帧槽 → RKNN NPU推理 → 算法后处理
            → 透明文字HUD → DRM/KMS双缓冲翻页 → HDMI
```

C++负责采集、推理、跟踪、图像处理和显示；Python负责启停、互斥、遥测与设备恢复。待处理帧和结果采用容量为一的最新帧策略，避免慢模型导致持续积压；主动替换次数单独统计。

HDMI画面左上角显示处理帧率、推理耗时、内部帧龄、CPU占用、温度、RSS及NPU负载。文字使用描边，无黑色背景。

当前适配器包含CPU映射、颜色转换和显式拷贝；`--capture-memory dmabuf`只改变采集内存模式，不代表整条链路实现零拷贝。

## 2026-11-25 第五轮：深度加速与三核吞吐

16:9相机默认采用518×294 FP16，保留原学习权重和实际518×291 RGB采样，移除原方形输入的上下空白。低延迟模式约3.55FPS、内部帧龄P95约366ms；三核并发约6.01FPS、P95约542ms，各核整轮平均约67%。原模式约1.41FPS、P95约846ms。单上下文完成180秒短测，三上下文仅20秒短测，长期验收仍未完成。

新旧深度输出会有差异。50图验证FP16转换一致性，18张既有相机帧的新旧深度排序、误差及梯度比较通过工程门槛；未做多场景或米制精度验收。image/video默认使用原518×518参考模型；非16:9实时输入需显式选择standard。

使用`bash depth_anything_v2/start_low_latency.sh`或`bash depth_anything_v2/start_throughput.sh`，切换前先停止。完整数据、限制和回退命令见[第五轮报告](docs/第五轮宽屏深度与三核调度优化.md)与[验证收据](docs/round5-validation.json)。

## 2026-11-25 更新：减少保存导致的HDMI停顿

已将图像、深度和张量保存移到有界后台线程。1080p30深度实测，`save_every=10`交替对比：首次显示帧龄约2682→824ms，翻页间隔P95约2592→741ms，稳态处理1.287→1.408FPS。模型调用仍约710ms；默认保存间隔300时，收益主要是首次和保存时的长尾改善。

深度180秒复测、YOLO与超分回归、固定图片输出一致性、真实保存失败和手动停止恢复均已检查。8组深度NPU候选未取得足够收益，保留原518×518 FP16。长期验收仍未完成。[第四轮完整报告](docs/第四轮深度候选与验证.md) · [验证收据](docs/round4-validation.json)。

三个推理项目可用`--save-every 0`关闭图像/张量保存。开启保存且磁盘繁忙时，只保留最新待写样本，跳过旧样本并统计；不适合作为完整录像接口。

## 实测性能

以下为 **2026-11-24 当前版本**在RK3588开发板上的实测。共同配置为单RKNN上下文、mask7、1080p30采样及1080p60 HDMI显示；模型输入尺寸见上表。

| 项目 | 稳态处理帧率 | 内部帧龄P95 | 请求测试时长 | 默认方案 |
|---|---:|---:|---:|---|
| YOLO11 + ByteTrack | 30.00FPS | 73.1ms | 20秒 | INT8检测 + CPU跟踪 |
| Real-ESRGAN | 11.81FPS | 175.4ms | 180秒 | FP16卷积图 + CPU尾部 |
| Depth Anything V2 Small | 1.42FPS | 1223.1ms | 20秒 | 原FP16模型 |

稳态FPS剔除前5秒；帧龄分位数覆盖该轮全部已处理帧，包含初始化期间的长尾。帧龄从appsink取样计到处理完成或HDMI翻页，不包含曝光、取样前等待和屏幕响应。短测结果不能代替长期验收，也不能视为光学端到端延迟。

### 第三轮NPU优化（历史基线）

逐层分析发现，原超分模型末端PixelShuffle被编译为单核反卷积，约占逐层耗时的31.5%。当前保留原权重和学习型卷积，直接读取FP16特征，将像素重排、原图残差相加和FP16舍入移到CPU/NEON完成。

同参数A/B交替对比结果（各两轮，每轮30秒）：

| 指标 | 原完整RKNN | 优化方案 |
|---|---:|---:|
| 稳态处理帧率 | 7.86FPS | **11.92FPS，提升约52%** |
| 模型调用P50 | 128.85ms | **83.15ms** |
| 内部帧龄P95 | 227.5ms | **175.0ms** |
| CPU占用，单核100%口径 | 65.1% | 79.1% |

表中分位数为每轮分位数的算术平均。固定50张验证图片上，优化方案与原FP16模型的最大浮点差、最大8bit像素差均为0。180秒复测最高温度为53.6℃；整板功耗未测。

深度模型的12个注意力算子约占逐层耗时的69%，主要集中在NPU核心0，部分Resize回退到CPU；第三轮完成了定位；第五轮已通过减少宽屏填充计算获得加速。三核掩码允许使用三核，实际并行程度仍由算子和编译图决定。

完整条件、逐层数据与限制见[第三轮NPU优化报告](docs/NPU瓶颈定位与第三轮优化.md)。

## 快速开始

### 1. 准备开发板

已验证环境：RK3588、8GB内存、ARM64 Linux、内核5.10.110、NPU驱动0.9.8、RKNN Runtime 2.3.2、OpenCV 4.2.0、GStreamer 1.16.2、libdrm 2.4.102。

开发板需要已有可用的IMX415驱动、ISP/IQ、RKAIQ服务和HDMI。当前适配使用`/dev/video44`、`/dev/v4l-subdev2`、`/dev/dri/card0`及DMA32堆；其他BSP需要重新核对设备节点和颜色格式。

以下命令在开发板的 **root终端**中执行。首次克隆使用尚不存在的目标目录；已有部署可直接进入工程目录进行启动操作。

```bash
mkdir -p /userdata/av_dev
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/guozhe010702-boop/new.git \
  "/userdata/av_dev/advanced algorithms"
cd "/userdata/av_dev/advanced algorithms"

# 首次缺少系统依赖时安装
bash tools/install_system_deps.sh

# Git克隆：拉取新宽屏模型的实际文件
git lfs install --local
git lfs pull --include="depth_anything_v2/models/fp16-camera-518x294.rknn"

# 环境预检；普通Git克隆不需要 --verify-package
python3 tools/preflight.py
```

仓库已包含该验证环境下的ARM64程序。修改源码或迁移系统环境后，先停止算法，再重新编译：

```bash
bash tools/build_board.sh
```

构建脚本同时执行CTest，并拒绝替换正在运行的本工程程序。工程使用`runtime/lib`中的私有RKNN/RGA库，系统依赖由开发板提供。

### 2. 启动、查看和停止

选择一个项目启动：

```bash
# 目标检测、跟踪与计数
bash yolo11_bytetrack/start.sh

# 或：图像超分
bash real_esrgan/start.sh

# 或：单目深度
bash depth_anything_v2/start.sh
```

切换项目之前先停止当前项目。三个项目共用状态与停止入口，以下命令也可停止当前运行的其他算法：

```bash
bash yolo11_bytetrack/status.sh
bash yolo11_bytetrack/stop.sh

# 查看服务日志
journalctl -u advanced-algorithms.service -n 80 --no-pager
```

启动会创建临时`advanced-algorithms.service`，关闭SSH窗口后程序仍会运行。停止脚本负责结束工作线程、释放资源并恢复原相机格式、桌面服务和显示布局；日常停止请使用脚本。

### 3. 有限时长测试

```bash
python3 tools/control.py run --project yolo11_bytetrack \
  --seconds 60 --output results/my-yolo-test
```

输出目录必须是新目录。实时采集默认独占相机和显示，不应与原MPP预览或其他相机程序同时运行。

## 常用操作

| 需求 | 参数或入口 |
|---|---|
| 跟踪人 / 所有COCO类别 | `--class 0` / `--class -1` |
| 水平计数线位置 | `--line 0.5`，归一化纵坐标 |
| 指定超分ROI左上角 | `--roi-x 100 --roi-y 100` |
| 单图片 / 视频文件 | `--image /userdata/test.jpg` / `--video /userdata/test.mp4` |
| 不输出HDMI画面 | `--hdmi 0` |
| 优化超分 / 原完整超分 | `--sr-tail cpu` / `--sr-tail npu` |
| 独立NPU阶段及逐层分析 | `tools/npu_benchmark.py` |

例如，显式比较两条超分路径：

```bash
python3 tools/control.py run --project real_esrgan --sr-tail npu \
  --seconds 60 --output results/my-sr-original
python3 tools/control.py run --project real_esrgan --sr-tail cpu \
  --seconds 60 --output results/my-sr-optimized
```

默认精度和调度从各项目的`recommended.json`读取。当前采用单上下文mask7，超分与深度保留FP16，YOLO使用INT8。超分和深度的既有INT8候选未通过相应画质门槛。

### 离线大图超分

```bash
LD_LIBRARY_PATH="$PWD/runtime/lib" ./build/sr_tiled \
  real_esrgan/models/fp16.rknn \
  /userdata/input-4k.png /userdata/output-x4.tif
```

该入口使用原完整超分模型。256×256分块、40像素halo、176像素有效步长，输入3840×2160可输出15360×8640。当前限制输入两边至少256且不超过4096像素。仅4K输入对应的×4 RGB输出数组就约占398MB；这是离线大图处理功能。

### 独立NPU分析

```bash
python3 tools/npu_benchmark.py --project depth_anything_v2 \
  --image /userdata/prepared-518.png --iterations 12 \
  --output results/my-depth-profile
```

分析输入需已准备为模型尺寸：深度518×518，超分256×256。该工具不打开摄像头；逐层采样与正常性能测试分开执行。

## 目录与核心程序

```text
common/                  C++采集、推理、后处理、显示和测试
depth_anything_v2/        深度模型、推荐配置、启停入口
real_esrgan/              超分模型、推荐配置、启停入口
yolo11_bytetrack/         检测模型、推荐配置、启停入口
runtime/                 私有RKNN/RGA运行库和头文件
vendor/                  第三方源码、来源及许可证
tools/                   监督控制、构建、转换、测量与验证
build/                   已测试的ARM64可执行文件
docs/                    流程、部署、实测报告与散列清单
results/                 完整本地实验资料、日志、张量及备份（大文件使用LFS）
data/                    原始相机采样、COCO128、准备后的校准/验证集（图片使用LFS）
```

| 程序 | 作用 |
|---|---|
| `common/main.cpp` | 实时程序入口、线程流水线、三算法后处理和HUD |
| `common/camera.cpp` | GStreamer采集、NV12转换及颜色格式处理 |
| `common/preprocess.cpp` | RGA颜色转换、缩放、letterbox和DMA32缓冲 |
| `common/model.cpp` | RKNN上下文、张量检查、输出生命周期和YOLO解码 |
| `common/sr_tail.hpp` | FP16特征直读、NEON像素重排与原图残差重建 |
| `common/display.cpp` | DRM/KMS资源管理、双缓冲和翻页事件 |
| `common/sr_tiled.cpp` | 离线整图分块超分 |
| `tools/control.py` | 手动启停、资源遥测、互斥和设备恢复 |

## 输出数据与模型重建

每次运行生成独立结果目录。`frames.csv`保存逐帧耗时；`tracks.csv`、`events.csv`保存轨迹和计数事件；`summary.json`、`analysis.json`保存统计；`telemetry.jsonl`保存资源采样；`restoration.json`保存恢复结果。选定帧还可保存原图、预览、超分图和浮点深度。

仓库已提供4个部署模型：YOLO INT8、深度FP16、超分FP16卷积图及原完整超分FP16模型。默认运行不需要重新转换。原始权重可用`tools/fetch_models.py`按固定SHA256下载；ONNX/RKNN转换分别由`tools/convert.py`、`tools/convert_sr_trunk.py`完成。

当前转换记录基于已有x86_64 Docker镜像`uav-yolo11-converter:2.3.2`；该镜像未随本仓库发布。重新转换需要准备对应环境，INT8校准需要代表性数据。Windows中已保存的相机图片、校准/验证数据、原始浮点张量及完整日志已使用Git LFS同步，清单见[完整资料清单](docs/full-data-manifest.json)，下载方法见[results说明](results/README.md)。

## 相机图片、数据集与完整日志

已同步Windows中的data/和results/，共2775个文件，约3.22GiB。图片、浮点张量、压缩包及日志等使用Git LFS，小型JSON/CSV仍可直接浏览。完整清单包含每个文件的SHA256；本次开发板SSH不可达，板端独有文件尚未核对。

快速开始默认跳过大文件下载，仍可运行相机实时推理。需要图片或原始实验资料时，安装Git LFS后执行：

```bash
git lfs install
git lfs pull

# 按需仅下载数据集或实验结果
git lfs pull --include="data/**"
git lfs pull --include="results/**"
```

普通git clone在启用LFS的环境会下载相应对象；网页或未启用LFS的下载方式可能只得到指针，需用git lfs pull补全。上传原始数据不改变既有测试结论，实验中断和未完成验收状态继续保留。

## 验证范围

- 当前相机链路按1080p30采样，HDMI验证模式为1080p60；模型处理的是各自固定尺寸输入。
- 单目深度输出是相对深度，超分生成细节未经过物理真实性验证；这些结果不能直接替代米制标定或科学测量。
- 超分180秒及其他项目短时回归已完成。此前完整验收在超分三上下文测试期间发生意外重启，原因仍未查明；桌面恢复附近仍有RGA映射异常。
- 当前版本尚未完成三项目长期稳定性验收，也未达到深度20FPS、超分25FPS级处理等初始目标。各项目`recommended.json`分别记录性能与稳定性状态。

## 详细文档

| 文档 | 内容 |
|---|---|
| [处理流程与接口](docs/处理流程与接口.md) | 帧、缓冲、模型接口、后处理及文件作用 |
| [部署操作](docs/部署操作.md) | 部署包迁移、手动控制、测试和恢复 |
| [NPU瓶颈定位与第三轮优化](docs/NPU瓶颈定位与第三轮优化.md) | 算子热点、等价图拆分、精度和性能对比 |
| [第二轮优化报告](docs/第二轮优化报告.md) | RGB向量化、显示缓存复用及A/B记录 |
| [板端实测报告](docs/实测报告.md) | 原验收记录、量化质量和验证边界 |
| [来源与发布清单](docs/github-publication.json) | 本次公开范围及运行资产散列 |

第三方模型和源码的许可证、来源记录分别保留在`vendor/`及`docs/model-sources.json`中；使用和分发时请遵循对应项目条款。
