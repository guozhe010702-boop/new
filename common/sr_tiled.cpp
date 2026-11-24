#include "core.hpp"
#include <filesystem>
#include <iostream>
using namespace aa;
int main(int argc,char**argv){try{
 require(argc==4,"Usage: sr_tiled MODEL INPUT_IMAGE OUTPUT_TIFF");require(!std::filesystem::exists(argv[3]),"output exists");Model model(argv[1],7);require(model.width==256&&model.height==256,"fixed 256 input required");cv::Mat bgr=cv::imread(argv[2]);require(!bgr.empty(),"image missing");require(bgr.cols<=4096&&bgr.rows<=4096,"maximum input dimensions 4096x4096");cv::Mat rgb;cv::cvtColor(bgr,rgb,cv::COLOR_BGR2RGB);
 // 34 conv layers -> radius 34; use a 40-pixel halo. Edge tiles are
 // shifted to the actual image boundary so per-layer zero padding is preserved.
 constexpr int halo=40,core=176;cv::Mat result(rgb.rows*4,rgb.cols*4,CV_8UC3);uint64_t tiles=0;int64_t start=now_us();
 require(rgb.cols>=256&&rgb.rows>=256,"tiled mode requires dimensions >=256");
 for(int y=0;y<rgb.rows;y+=core)for(int x=0;x<rgb.cols;x+=core){int cw=std::min(core,rgb.cols-x),ch=std::min(core,rgb.rows-y);int x0=std::clamp(x-halo,0,rgb.cols-256),y0=std::clamp(y-halo,0,rgb.rows-256);auto tile=rgb(cv::Rect(x0,y0,256,256)).clone();auto o=model.run(tile);require(o.size()==1&&o[0].size()==3*1024*1024,"SR tensor shape");for(int yy=0;yy<ch*4;++yy){auto dst=result.ptr<cv::Vec3b>(y*4+yy)+x*4;for(int xx=0;xx<cw*4;++xx)for(int c=0;c<3;++c)dst[xx][c]=cv::saturate_cast<uint8_t>(255*o[0][c*1024*1024+(yy+(y-y0)*4)*1024+xx+(x-x0)*4]);}++tiles;}
 cv::cvtColor(result,bgr,cv::COLOR_RGB2BGR);require(cv::imwrite(argv[3],bgr),"save TIFF failed");std::ofstream f(std::string(argv[3])+".json");f<<"{\"input_width\":"<<rgb.cols<<",\"input_height\":"<<rgb.rows<<",\"output_width\":"<<result.cols<<",\"output_height\":"<<result.rows<<",\"tile\":256,\"halo\":40,\"tiles\":"<<tiles<<",\"seconds\":"<<(now_us()-start)/1000000.<<",\"output_image_bytes\":"<<result.total()*3<<"}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
