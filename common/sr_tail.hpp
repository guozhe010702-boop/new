#pragma once
#include "core.hpp"
#include <type_traits>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace aa {
// PixelShuffle(CRD,4) + nearest-neighbor input residual, preserving FP16 output.
// Learned convolutions stay in the RKNN trunk. This is not a new SR network.
template<class T> inline std::vector<float> sr_cpu_tail_data(const T* trunk,size_t count,const cv::Mat& rgb){
 require(rgb.type()==CV_8UC3&&rgb.cols==256&&rgb.rows==256&&count==size_t(48*256*256),"SR CPU tail contract");
 cv::Mat normalized,half;rgb.convertTo(normalized,CV_32F,1./255);cv::convertFp16(normalized,half);cv::convertFp16(half,normalized);std::vector<cv::Mat> channels;cv::split(normalized,channels);
 std::vector<float> result(3*1024*1024);
 for(int c=0;c<3;++c)for(int y=0;y<256;++y)for(int dy=0;dy<4;++dy){
  const float* base=channels[c].ptr<float>(y);const T* rows[4];for(int dx=0;dx<4;++dx)rows[dx]=trunk+(c*16+dy*4+dx)*256*256+y*256;float* dst=result.data()+c*1024*1024+(y*4+dy)*1024;
#if defined(__aarch64__)
  for(int x=0;x<256;x+=4){auto residual=vld1q_f32(base+x);float32x4x4_t values;for(int dx=0;dx<4;++dx){float32x4_t features;if constexpr(std::is_same<T,float>::value)features=vld1q_f32(rows[dx]+x);else features=vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(rows[dx]+x)));values.val[dx]=vcvt_f32_f16(vcvt_f16_f32(vaddq_f32(features,residual)));}vst4q_f32(dst+x*4,values);}
#else
  for(int x=0;x<256;++x)for(int dx=0;dx<4;++dx){float feature;if constexpr(std::is_same<T,float>::value)feature=rows[dx][x];else feature=float(cv::float16_t::fromBits(rows[dx][x]));dst[x*4+dx]=float(cv::float16_t(feature+base[x]));}
#endif
 }
 return result;
}
inline std::vector<float> sr_cpu_tail(const std::vector<float>& trunk,const cv::Mat& rgb){return sr_cpu_tail_data(trunk.data(),trunk.size(),rgb);}
inline std::vector<float> sr_cpu_tail_fp16(const void* trunk,size_t count,const cv::Mat& rgb){return sr_cpu_tail_data(static_cast<const uint16_t*>(trunk),count,rgb);}
}
