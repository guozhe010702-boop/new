"""Generate the bounded optimization report from archived board receipts."""
from pathlib import Path
import json,statistics

ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'results/optimization-v2'
    job=json.loads((folder/'job.json').read_text(encoding='utf-8'))
    final=json.loads((folder/'final-state.json').read_text(encoding='utf-8'))
    assert job['stage']=='complete' and final['same_boot']
    cases={c['name']:c['analysis'] for c in job['cases']}
    a=[cases[n] for n in ('a1','a2')];b=[cases[n] for n in ('b1','b2')]
    def avg(group,key,sub=None):return statistics.mean(x[key][sub] if sub else x[key] for x in group)
    fps_a=avg(a,'steady_processed_fps');fps_b=avg(b,'steady_processed_fps')
    comparisons=[('处理FPS',fps_a,fps_b),('后处理P50 / ms',avg(a,'postprocess_ms','p50'),avg(b,'postprocess_ms','p50')),('帧龄P95 / ms',avg(a,'age_ms','p95'),avg(b,'age_ms','p95')),('CPU / 单核百分比',avg(a,'native_cpu_one_core_percent'),avg(b,'native_cpu_one_core_percent'))]
    rows=['# 第二轮优化与有界复测（2026-11-24）','',
          '本轮已完成代码修改、板端编译、3项CTest、A/B交替测试和三项目功能回归。原完整验收因开发板意外重启中断，尚未恢复长测；本报告不表示三项目已完成稳定性验收。','',
          '## 修改与作用','',
          '- `common/render.hpp`：将超分输出的三个FP32平面分别批量转换为8bit，再合并RGB。保留逐像素标量实现，用于独立对照；模型、分辨率和精度未改变。',
          '- `common/main.cpp`：超分后处理使用上述向量路径；`--render scalar|vector`可切换，默认为vector。',
          '- `common/display.cpp`：每个DRM缓冲只在显示内容尺寸变化时重置背景；复用缩放缓冲；尺寸已匹配时直接转换颜色，省去一次相同尺寸的resize。继续保持画面比例和真实翻页事件计时。',
          '- `tools/control.py`：遥测文件首次采样及之后每5秒执行fsync，增加异常掉线前记录的可用性；不能保证掉电前最后5秒或所有内核日志都能保留。',
          '- `common/render_tests.cpp`：固定随机种子，覆盖小于0、大于1和正常范围的1024×1024 RGB张量，检查标量/向量结果与实际板端耗时。','',
          '## 正确性与独立转换测试','',
          '板端CTest：core、RGA颜色/stride、render共3/3通过，原始日志见results/optimization-v2/ctest.log。1024×1024张量转换20轮平均：标量28.4697ms，向量2.7733ms，约10.3倍；本次样本的最大像素差为0。测试容许至多1级舍入差，不代表所有输入数学上逐位一致。该结果只描述RGB转换，不代表整条超分链路加速10倍。','',
          '## 同参数A/B交替对比','',
          'A为归档旧可执行文件，B为本轮优化可执行文件。顺序A1→B1→B2→A2，每轮40秒，同一相机场景但非逐帧锁定录像。均为Real-ESRGAN FP16、256×256 ROI→1024×1024、一RKNN上下文、mask7、1080p30采集及1080p60 HDMI显示；禁用图片保存。处理FPS剔除前5秒，阶段分位数来自该轮全部已处理帧。','',
          '| 指标 | A两轮均值 | B两轮均值 | 相对变化 |','|---|---:|---:|---:|']
    for label,x,y in comparisons:rows.append(f'| {label} | {x:.3f} | {y:.3f} | {(y/x-1)*100:+.2f}% |')
    rows+=['','以上P50/P95是每轮分位数的算术平均，不是合并所有样本后的分位数。CPU按原生进程单核100%统计，CPU下降不等于同幅度整板功耗下降。短时A/B存在温度、相机内容和调度波动，不能视为统计显著性证明。','',
           '| 轮次 | FPS | 后处理P50 ms | 帧龄P95 ms | 显示P50 ms | CPU单核% |','|---|---:|---:|---:|---:|---:|']
    for n in ('a1','b1','b2','a2'):
        v=cases[n];rows.append(f"| {n} | {v['steady_processed_fps']:.3f} | {v['postprocess_ms']['p50']:.1f} | {v['age_ms']['p95']:.1f} | {v['display_ms']['p50']:.1f} | {v['native_cpu_one_core_percent']:.2f} |")
    rows+=['','## 新版独立复测','',
           '| 项目 | 请求时长 | 配置 | 稳态FPS | 推理P50 ms | 帧龄P95 ms | 最高温度℃ | 恢复 |','|---|---:|---|---:|---:|---:|---:|---|']
    for n,duration,config in [('stability',180,'超分FP16，1上下文 mask7'),('yolo-regression',30,'YOLO INT8，1上下文 mask7'),('depth-regression',30,'深度FP16，1上下文 mask7')]:
        v=cases[n];rows.append(f"| {config} | {duration}秒 | 1080p30采集 / HDMI | {v['steady_processed_fps']:.3f} | {v['inference_ms']['p50']:.1f} | {v['age_ms']['p95']:.1f} | {v['max_temperature_c']:.2f} | {'通过' if v['restoration_pass'] else '失败'} |")
    sr=cases['stability'];depth=cases['depth-regression']
    rows+=['',f"超分180秒请求实际运行{sr['seconds']:.2f}秒（含初始化与退出），已处理{sr['processed']}帧；采集待处理旧帧主动替换{sr['raw_replaced']}次、结果候选替换{sr['results_replaced']}次、乱序拒绝{sr['out_of_order_results']}次，错误标志false，RSS最大{sr['rss_max_mib']:.2f}MiB。这些替换用于避免积压，不计为完成的推理帧。短测不能证明长期无泄漏。",'',
           f"深度此次推理P50为{depth['inference_ms']['p50']:.1f}ms，后处理P99为{depth['postprocess_ms']['p99']:.1f}ms，启动期存在明显长尾。仅约{depth['processed']}个完成样本，不能用此短测证明稳定性或20FPS达标。",'',
           '帧龄从appsink取样开始，到HDMI翻页/处理完成为止；不包括曝光、传感器读取或取样前积压。以上不是光学端到端延迟。','',
           '## 重启调查与保守默认值','',
           '上一轮原验收在超分三上下文第二次300秒测试期间意外重启。用户确认没有操作供电或重启。旧启动ID为ee411906-ff53-4e09-a4da-86d9db199a65，新启动ID为033342b6-8591-4477-b0f4-4099656b5f2b；系统日志显示12:27:24前后停止记录，12:27:55开始新启动。未找到留存的panic，pstore为空，具体原因尚未确定。',
           '', 'RGA日志有超过4GB映射错误，但旧启动早期、算法验收前就存在同类错误，不能据此归因本程序。本轮内核日志另外记录14条同类错误，分别集中在7次restored_at前后两秒内，每次2条，显示与桌面恢复过程存在时间相关；还没有按发起进程完成归因。这属于未解决的驱动/显示恢复异常，不因应用错误标志false而认定系统层无错误。时间对照见results/optimization-v2/rga-error-timing.json，完整内核区间日志也已保留。',
           '', '没有擅自修改内核、RGA核心调度、供电或频率。本轮连续启动ID保持不变，三项目逐个运行，退出后/dev/video44无应用占用，gdm3恢复active，算法与原验收服务均inactive。内核另有rkcif dummy-buffer及未连接sensor通道日志，当前没有证明整个摄像头驱动和ISP都进入低功耗状态。',
           '', '三项目默认均采用单上下文mask7。YOLO保持INT8，超分与深度保持已验证的FP16。当前版本recommended.json明确标记长时间稳定性未完成；此前YOLO三上下文1800秒29.987FPS的通过记录保留为旧版本证据，不能直接移植为本轮可执行文件的长测结论。','',
           '## 量化实验状态','',
           '额外在Windows Docker内进行MMSE INT8校准搜索，使用原校准集均匀选出的32张图片，不占开发板NPU。约5分钟后仍只完成3/39层，工具估计剩余超过1小时，本轮主动结束该实验并保留日志。没有导出候选RKNN、没有做候选精度验收，也未认定MMSE量化精度失败。既有普通INT8超分SSIM 0.9052未达到0.98门槛，默认仍FP16。实验记录见real_esrgan/experiments/mmse32/conversion.json。','',
           '## 当前瓶颈与下一步优先级','',
           f"1. 超分推理P50 {sr['inference_ms']['p50']:.1f}ms，已明显高于后处理{sr['postprocess_ms']['p50']:.1f}ms；向量化主要释放CPU并缩短帧龄。显示P50仍约{sr['display_ms']['p50']:.1f}ms，没有证明翻页等待显著降低。",
           '2. 深度应先单独分析RKNN图中的算子分配、转换和数据搬运，保留518输入和画质基线；只有候选模型在同一验证集通过精度门槛后再比较速度。不要直接降低输入尺寸或换网络冒充本模型优化。',
           '3. 重启原因未明前保持有界单上下文运行。后续定位应结合串口持续内核日志、供电测量和分阶段负载复现；当前数据不足以断言供电、温度或驱动中的某项是原因。',
           '4. 量化/混合量化属于后续候选实验，需验证板端能力、相同输入、画质及长时间恢复。当前不把未验证候选设成默认。','',
           '## 文件与手动操作','',
           '源代码与可执行文件已同步Windows和开发板。默认不开机自启，本轮结束后实时预览保持停止。','',
           '```bash','cd "/userdata/av_dev/advanced algorithms"','bash real_esrgan/start.sh  # 也可选择其他项目','bash real_esrgan/status.sh','bash real_esrgan/stop.sh','',
           '# 有界前台测试，输出目录每次取新名字','python3 tools/control.py run --project real_esrgan --seconds 60 --workers 1 --mask 7 --render vector --save-every 0 --output results/my-vector-test','```','',
           '原始数据：results/optimization-v2/job.json及各轮frames.csv、analysis.json、telemetry.jsonl、restoration.json；最终状态：results/optimization-v2/final-state.json；重启调查：results/reboot-investigation/。可用python3 tools/optimization_v2_report.py重新生成本文。','',
           f"本轮二进制SHA256：`{final['binary_sha256']}`。",
           f"A/B旧基线SHA256：`{final['baseline_binary_sha256']}`。",'']
    (ROOT/'docs/第二轮优化报告.md').write_text('\n'.join(rows),encoding='utf-8')
    print('Optimization report generated from seven completed cases.')

if __name__=='__main__':main()
