# 第三轮：NPU瓶颈定位与超分图优化

日期：2026-11-24。已完成两模型的独立API分段/逐层分析、超分FP16等价图拆分、50张固定验证集对比、摄像头到HDMI交替对比，以及180秒超分连续复测。深度模型本轮完成瓶颈定位，保持原模型实现。未完成30分钟稳定性验收，未解决此前意外重启的根因。

## 测量条件与边界

RK3588、NPU驱动0.9.8、项目私有RKNN Runtime/Toolkit2 2.3.2。采样时NPU为1GHz、DMC为2.112GHz，两者governor均为performance；本轮没有改频率、调压或更换驱动。每次只有一个推理任务，单上下文mask7，保留原输入尺寸与已验证FP16精度。

独立分析使用已准备的固定图片，绕过摄像头和显示。正常API耗时关闭COLLECT_PERF；逐层详情单独启用该标志。官方API明确说明逐层采样存在性能开销，故不把详情表当作正式吞吐测试：[Rockchip RKNN API](https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h)。

## 找到的具体热点

| 模型 | 算子 | 调用数 | CPU耗时 ms | NPU耗时 ms | 逐层时间占比 |
|---|---|---:|---:|---:|---:|
| 超分 | ConvPRelu | 33 | 0.000 | 66.044 | 61.92% |
| 超分 | ConvTranspose | 1 | 0.000 | 33.606 | 31.51% |
| 超分 | Add | 1 | 0.000 | 2.174 | 2.04% |
| 超分 | Resize | 1 | 0.000 | 1.866 | 1.75% |
| 深度 | exSDPAttention | 12 | 0.000 | 445.369 | 69.16% |
| 深度 | Resize | 5 | 64.966 | 0.093 | 10.10% |
| 深度 | ConvExGelu | 12 | 0.000 | 33.568 | 5.21% |
| 深度 | exNorm | 28 | 0.000 | 29.404 | 4.57% |

超分末端PixelShuffle被编译为ConvTranspose，详情表显示其工作量集中在核心0，用时33.606ms、占31.51%；33个ConvPRelu合计66.044ms。深度的12个exSDPAttention共445.369ms、占69.16%，其工作量也集中在核心0；Resize存在CPU回退。mask7允许三核使用，实际算子调度仍受编译图和实现限制。

## 采用的超分优化

保留Real-ESRGAN general-x4v3原始权重和全部学习型卷积，RKNN只输出最后一层卷积的48×256×256特征。把PixelShuffle×4、nearest原图残差相加及FP16舍入改为CPU端完成。输入仍为256×256 RGB，逻辑输出仍为3×1024×1024。

对通道c、输入坐标h,w、子像素i,j，重建关系为：

```text
Y[c, 4h+i, 4w+j] = FP16(T[16c+4i+j, h, w] + FP16(X[h,w,c]/255))
i,j ∈ {0,1,2,3}
```

C++直接请求逻辑NCHW FP16输出（want_float=0），在SDK输出有效期内读取；NEON批量扩展、相加、舍入并交错存储为逻辑FP32 RGB。省去运行时的整张FP32特征转换及中间向量拷贝。输出读取结束后才调用rknn_outputs_release，保留SDK缓存一致性行为。本路径仍有输出整理和CPU访问，不宣称DMA-BUF全链路零拷贝。

## 正确性证据

- 提取ONNX时逐一检查保留下来的initializer字节不变；没有重训练或替换网络。
- 两组固定随机输入验证“完整ONNX”与“卷积ONNX＋重建公式”，最大绝对差均为0。
- 板端固定50张验证图片：最大浮点差0、最大8bit像素差0；最终FP16直读版本与原完整FP16 RKNN逐元素一致。零MSE时程序把PSNR封顶为200dB；应以差值为0理解，不将该封顶数字当作独立画质提升。
- 验证图按data/dataset.json的prepared_sha256核对；质量测试是与既有FP16基线对齐，不证明所有可能输入永远逐位一致，也不证明生成细节的真实性。
- 板端CTest三项通过，新增CPU尾部另以原始完整RKNN的真实输出作为独立参照，不用同一份重建实现自证正确。

## 独立调用分段耗时

下表为正常模式；基线是A/B两轮各12次的中位数均值，优化版是30次的中位数。预热均为3次。同一图片，单上下文mask7；这些是不同测试轮的统计值，不等同于逐帧配对或同一个总样本分位数。

| 阶段 | 原完整RKNN ms | 优化方案 ms |
|---|---:|---:|
| 输入提交 | 1.595 | 1.613 |
| 执行等待 | 105.425 | 67.344 |
| 输出获取 | 11.136 | 3.689 |
| 额外特征拷贝 | 4.970 | 0.000 |
| CPU重建 | 0.000 | 5.434 |
| 总调用 | 123.619 | 78.078 |

## 摄像头到HDMI交替对比

A=完整NPU尾部，B=FP16直读＋CPU尾部，使用同一个当前可执行文件。顺序A1→B1→B2→A2，各请求30秒，单上下文mask7、FP16、同一相机场景、1080p30采样及1080p60显示。固定256 ROI，保存策略相同；每轮首张图片仍保存，steady FPS剔除前5秒。相机内容没有逐帧锁定。

| 指标 | A两轮均值 | B两轮均值 | 相对变化 |
|---|---:|---:|---:|
| 处理FPS | 7.864 | 11.922 | +51.61% |
| 调用P50 ms | 128.850 | 83.150 | -35.47% |
| 内部帧龄P95 ms | 227.500 | 174.950 | -23.10% |
| CPU单核% | 65.101 | 79.060 | +21.44% |

阶段分位数覆盖本轮全部已处理帧；表中P50/P95是每轮分位数的均值。CPU单核100%口径，优化后部分运算移至CPU，CPU占用上升；整板功耗未测，不能由帧率或CPU百分比推算节能。内部帧龄从appsink取样到翻页，不含曝光、取样前等待或屏幕响应。

| 轮次 | 处理FPS | 调用P50 ms | 帧龄P95 ms | CPU单核% |
|---|---:|---:|---:|---:|
| a1 | 7.884 | 129.0 | 232.1 | 64.60 |
| b1 | 11.894 | 83.2 | 175.7 | 78.90 |
| b2 | 11.950 | 83.1 | 174.2 | 79.22 |
| a2 | 7.844 | 128.7 | 222.9 | 65.60 |

## 连续复测与当前限制

| 项目 | 请求时长 s | 稳态FPS | 帧龄P95 ms | 温度最大℃ | 退出恢复 |
|---|---:|---:|---:|---:|---|
| cpu-stability | 180 | 11.814 | 175.4 | 53.62 | 通过 |
| yolo-regression | 20 | 29.998 | 73.1 | 49.92 | 通过 |
| depth-regression | 20 | 1.420 | 1223.1 | 50.85 | 通过 |

CPU尾部连续复测实际运行182.35s，完成2123帧；主动替换采集候选3269次、结果候选3次，乱序拒绝0次，应用错误标志false，RSS最大216.32MiB。替换不算作完成的推理，短测不证明长期无泄漏。

当前超分仍未达到最初40ms纯推理参考或25FPS级处理，深度也未达到20FPS。原意外重启原因仍未查明，桌面恢复附近的RGA映射异常仍需定位；本轮无重启不能代替完整稳定性验收。默认保持单上下文，不恢复三上下文后台长测，结束后预览停止。

## 未采用的候选

普通preallocated FP32输出在超分实测总调用约137ms，比原约124ms更慢，因此未启用。RKNN_FLAG_ENABLE_SRAM的内存查询total_sram_size和free_sram_size均为0，当前BSP没有向该上下文提供可用SRAM；未把“API接收标志”当作实际SRAM加速。没有更改模型精度、输入尺寸、缓存刷新规则或使用返回上一帧结果的异步模式。

## 文件与操作

- common/npu_profile.cpp：独立RKNN阶段、原生张量属性和逐层详情诊断。
- common/sr_tail.hpp：PixelShuffle＋残差重建及FP16/NEON处理。
- common/model.cpp：FP16输出生命周期和CPU尾部可选路径；其他模型保留原路径。
- common/sr_tail_check.cpp：用完整RKNN逐图验证候选，输出质量CSV和预览。
- tools/convert_sr_trunk.py：原始ONNX提取、等价验证及FP16 RKNN转换。
- tools/npu_benchmark.py：有界、独占锁、无摄像头的可重复诊断。
- real_esrgan/models/fp16-trunk.rknn：优化候选；保留fp16.rknn供原方案对照。
- results/npu-bottleneck-v1/：原始阶段CSV、逐层表、质量记录及live-abba复测；before-cpu-tail.tgz在开发板保留上一版源码与二进制。

```bash
cd "/userdata/av_dev/advanced algorithms"
# 默认使用已通过本轮短测的CPU尾部；仍是FP16、单上下文
bash real_esrgan/start.sh
bash real_esrgan/stop.sh

# 显式对比原方案，输出目录每次使用新名字
python3 tools/control.py run --project real_esrgan --sr-tail npu --seconds 60 --output results/my-original-test
python3 tools/control.py run --project real_esrgan --sr-tail cpu --seconds 60 --output results/my-cpu-tail-test

# 独立诊断，输入必须已准备成对应模型尺寸
python3 tools/npu_benchmark.py --project depth_anything_v2 --iterations 12 --output results/my-depth-profile
# 若当前部署未带验证集，可加 --image /userdata/prepared-518.png
```

下一步深度优先分析单核exSDPAttention的编译拆分/多核实现，再研究CPU Resize回退；采用同权重、同518输入、同验证集作比较。当前还没有实测可用的深度图优化，保持原FP16模型。

当前完整链路可执行文件SHA256：`ea46597445e209b4ce30f42a933fe9d87ff0a2e8798918631548e2dfb13c466b`。
