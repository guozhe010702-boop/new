#include "core.hpp"
#include <iostream>
using namespace aa;
int main(){try{
 RgaResize rga;constexpr int w=640,h=360;
 for(auto values:std::vector<std::array<int,6>>{{0,128,128,0,0,0},{255,128,128,255,255,255},{54,99,255,254,0,0}}){
  cv::Mat nv(h*3/2,w,CV_8U);nv.rowRange(0,h).setTo(values[0]);for(int y=h;y<h*3/2;++y)for(int x=0;x<w;x+=2){nv.at<uint8_t>(y,x)=values[1];nv.at<uint8_t>(y,x+1)=values[2];}
  auto rgb=rga.nv12_rgb(nv,w,h);auto mean=cv::mean(rgb);for(int c=0;c<3;++c)require(std::abs(mean[c]-values[c+3])<=4,"BT709 full-range color mismatch");
 }
 cv::Mat rgb(h,w,CV_8UC3,cv::Scalar(200,50,20));auto resized=rga.resize(rgb,518,292);require(resized.cols==518&&resized.rows==292&&resized.isContinuous(),"RGA aligned stride unpack");auto v=resized.at<cv::Vec3b>(100,100);require(std::abs(v[0]-200)<=1&&std::abs(v[1]-50)<=1,"RGA RGB ordering");
 auto lb=rga.letterbox(rgb,518);require(lb.rgb.cols==518&&lb.x==0&&lb.y>0,"letterbox geometry");
 cv::Mat red(h*3/2,w,CV_8U);red.rowRange(0,h).setTo(67);for(int y=h;y<h*3/2;++y)for(int x=0;x<w;x+=2){red.at<uint8_t>(y,x)=92;red.at<uint8_t>(y,x+1)=255;}auto bt2020=cpu_nv12_rgb(red,w,h,.2627,.0593,true);auto mean=cv::mean(bt2020);require(mean[0]>=250&&mean[1]<=4&&mean[2]<=4,"CPU BT2020 full range red");
 std::cout<<"RGA color, stride518, letterbox and CPU BT2020 passed\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
