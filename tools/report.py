#!/usr/bin/env python3
"""Generate a report from measured receipts; an unfinished job stays unfinished."""
from pathlib import Path
import json,datetime
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
def main():
    job=read(ROOT/'results/qualification-20261124/job.json');rows=['# 板端实测报告','','报告根据实际转换、板端输出与验收记录生成。三个模型独立运行，不把模型推理耗时当作整条链路帧率。','','## 验收进度','']
    optimization=read(ROOT/'results/optimization-v2/job.json');npu=read(ROOT/'results/npu-bottleneck-v1/live-abba/job.json')
    if npu and npu.get('stage')=='complete':rows+=['最新NPU瓶颈定位与超分图优化已完成，详见[NPU第三轮报告](NPU瓶颈定位与第三轮优化.md)。下面保留原完整验收数据，当前版本未完成三项目长时间稳定性验收。','']
    elif optimization and optimization.get('stage')=='complete':rows+=['本轮代码优化与有界复测已完成，详见[第二轮优化报告](第二轮优化报告.md)。下面保留原完整验收数据，当前版本未完成三项目长时间稳定性验收。','']
    if job:
        rows+=['当前状态：**'+job['stage']+'**。已记录 '+str(len(job['cases']))+' 个测试项。','', '每个模型的规定验收为三轮各300秒和一次1800秒连续运行，另有调度扫描及阻塞测试。只有完整时长结束后才确认稳定性。','']
        if job['stage']=='running':rows+=['正在运行：`'+job.get('current','')+'`。长测尚未结束，以下已完成数据不代替最终验收。','']
        if job['stage']=='interrupted_reboot':rows+=['原验收在超分测试期间因开发板意外重启中断，用户确认没有操作开发板；原因尚未查明。当前已停止原后台验收，不会自动恢复高负载测试。','']
        if job.get('error'):rows+=['失败或中断原因：'+job['error'],'']
    else:rows+=['完整串行验收记录尚未同步到本目录，不宣称验收完成。','']
    delivery=read(ROOT/'results/async-delivery.json');audit=read(ROOT/'results/final-audit.json');post=read(ROOT/'results/post-qualification.json')
    if delivery and delivery.get('stage')=='failed':rows+=['自动交付未通过：'+delivery.get('error','未知错误')+'。不能据此宣称最终验收通过。','']
    if audit:rows+=['最终资源与帧序审计：'+('通过' if audit.get('passed') else '尚未通过')+'。','']
    if post:rows+=['退出恢复及最终版本复测：'+post.get('stage','未知')+'。','']
    rows+=['## 模型精度与量化','', '| 项目 | 板端FP16与ONNX对齐 | INT8结果 | 默认精度 |','|---|---|---|---|']
    for name,label in [('yolo11_bytetrack','YOLO11n'),('real_esrgan','Real ESRGAN'),('depth_anything_v2','Depth Anything V2 Small')]:
        q=read(ROOT/'results'/(name+'-quality.json'))
        if not q:rows.append('| '+label+' | 未完成 | 未完成 | 待定 |');continue
        if name=='yolo11_bytetrack':baseline='子集mAP ONNX %.4f / FP16 %.4f'%(q.get('onnx_subset_map',0),q['fp16_subset_map']);quant='子集mAP %.4f'%q['int8_subset_map']
        elif name=='real_esrgan':baseline='平均PSNR %.2fdB'%q.get('fp16_onnx_mean_psnr_db',0);quant='PSNR %.2fdB SSIM %.4f'%(q['mean_psnr_db'],q['mean_ssim'])
        else:baseline='中位秩相关 %.5f'%q.get('fp16_onnx_median_spearman',0);quant='中位秩相关 %.4f'%q['median_spearman']
        rows.append('| '+label+' | '+baseline+' | '+quant+(' 通过' if q['int8_quality_pass'] else ' 未通过')+' | '+('INT8' if q['int8_quality_pass'] else 'FP16')+' |')
    rows+=['','校准194张，验证50张，按原图SHA256隔离。验证含32张COCO128图片和18张相机图片；YOLO的mAP只统计有标注的COCO子集。此划分隔离的是校准与验证，不声称模型未在这些COCO训练图片上训练过，也不代表完整COCO验证集精度。深度和超分指标用于量化损失对比，不是米制深度准确率或微观细节真实性证明。','','## 调度与持续测试','', '| 测试 | 时长秒 | 配置 | 处理FPS | 帧龄P95毫秒 | 最高温度 | 恢复 |','|---|---:|---|---:|---:|---:|---|']
    if job:
        for c in job['cases']:
            a=c.get('analysis',{});rows.append('| '+c['name']+' | '+str(c['seconds_requested'])+' | '+c['precision']+' '+str(c['workers'])+'上下文 mask'+str(c['mask'])+' | %.3f | %.1f | %.1f | %s |'%(a.get('steady_processed_fps',a.get('fps',0)),a.get('age_ms',{}).get('p95',0),a.get('max_temperature_c',0),'通过' if a.get('restoration_pass') else '未通过'))
    rows+=['','处理FPS按去除前5秒后的实际处理计数与采样时间计算。帧龄起点为appsink取样，包含后续颜色转换、排队、推理、后处理和显示等待；不包含曝光与采样前的时间。CPU按原生进程单核100%统计，RSS不包含全部NPU/DRM内核缓冲或桌面进程。','','## 功能与恢复','']
    life=read(ROOT/'results/lifecycle/result.json')
    if life:
        for c in life:rows.append('- '+c['name']+'：'+('通过' if c['passed'] else '失败'))
    else:rows.append('恢复测试记录尚未同步。')
    seam=read(ROOT/'results/seam-test/comparison.json')
    if seam:rows+=['','分块结果与384方形整图PyTorch参考的PSNR为%.2fdB，接缝带平均绝对误差为%.4f个8bit灰度级，检查%s。'%(seam['psnr_db'],seam['seam_band_mae_8bit'],'通过' if seam['passed'] else '失败')]
    tiled=read(ROOT/'results/4k-x4.tif.json')
    if tiled:rows+=['','真实相机4K输入3840×2160已输出15360×8640，共%d块，完整处理及保存耗时%.2f秒。输出RGB数组大小%d字节。'%(tiled['tiles'],tiled['seconds'],tiled['output_image_bytes'])]
    aspect=read(ROOT/'results/display-aspect-fix/change.json')
    if aspect:rows+=['','深度及超分的并排画面在YOLO长测结束后改为保持比例居中显示；输入1920×1080的YOLO路径继续全屏，推理与跟踪代码未改变。变更前后可执行文件散列见results/display-aspect-fix/change.json。']
    final=read(ROOT/'results/yolo-final-display-smoke/summary.json')
    if final:rows+=['','最终显示版本另做YOLO %.1f秒复测，平均处理FPS %.3f，错误标志%s。'%(final['seconds'],final['fps'],str(final['error']))]
    rows+=['','## 性能目标与限制','', '初始目标是深度20FPS、超分256方形ROI约40ms纯推理、YOLO跟踪30FPS级处理。上述目标不由示例数字自动保证。最终以各项目recommended.json中的measured_fps、stability_validated与performance_target_met分别说明实测帧率、稳定性和性能达标情况。','', '深度与超分的INT8在当前校准配置下损失过大，按约定保留FP16；不得把INT8速度与FP16画质混成一个结果。NPU三核上下文增加吞吐也可能增加单帧耗时。模型、尺寸与画质优先，未达目标时保留实测瓶颈。','', '物理HDMI断连、光学端到端延迟、米制深度标定和科学成像任务精度不属于本次已通过的声明。注入显示异常与真实拔线不同。','']
    (ROOT/'docs/实测报告.md').write_text('\n'.join(rows),encoding='utf-8')
if __name__=='__main__':main()
