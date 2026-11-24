#pragma once
#include "core.hpp"
namespace aa {
// The scalar path is retained as an independent reference for A/B validation.
inline cv::Mat rgb_from_planar(const std::vector<float>& data,int width,int height,bool vectorized=true){
 const size_t pixels=size_t(width)*height;require(data.size()==pixels*3,"RGB planar tensor size mismatch");
 cv::Mat rgb(height,width,CV_8UC3);
 if(!vectorized){for(int y=0;y<height;++y){auto dst=rgb.ptr<cv::Vec3b>(y);for(int x=0;x<width;++x)for(int c=0;c<3;++c)dst[x][c]=cv::saturate_cast<uint8_t>(255*data[c*pixels+size_t(y)*width+x]);}return rgb;}
 cv::Mat channels[3];
 for(int c=0;c<3;++c){cv::Mat plane(height,width,CV_32F,const_cast<float*>(data.data()+c*pixels));plane.convertTo(channels[c],CV_8U,255);}
 cv::merge(channels,3,rgb);return rgb;
}
}
