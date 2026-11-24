#include "core.hpp"
#include <filesystem>
#include <iostream>
#include <numeric>
namespace fs=std::filesystem;using namespace aa;
double ssim(const cv::Mat& a,const cv::Mat& b){
 cv::Mat u,v,aa,bb,ab;cv::GaussianBlur(a,u,{11,11},1.5);cv::GaussianBlur(b,v,{11,11},1.5);cv::GaussianBlur(a.mul(a),aa,{11,11},1.5);cv::GaussianBlur(b.mul(b),bb,{11,11},1.5);cv::GaussianBlur(a.mul(b),ab,{11,11},1.5);aa-=u.mul(u);bb-=v.mul(v);ab-=u.mul(v);cv::Mat numerator=(2*u.mul(v)+.0001).mul(2*ab+.0009),denominator=(u.mul(u)+v.mul(v)+.0001).mul(aa+bb+.0009),ratio;numerator.copyTo(ratio);cv::divide(numerator,denominator,ratio);return cv::mean(ratio(cv::Rect(5,5,a.cols-10,a.rows-10)))[0];
}
int main(int argc,char**argv){try{
 require(argc==5,"Usage: sr_tail_check FULL_MODEL TRUNK_MODEL PREPARED_VALIDATION_DIR NEW_OUTPUT");require(!fs::exists(argv[4]),"output must be new");fs::path out=argv[4];fs::create_directories(out);cv::setNumThreads(1);Model full(argv[1],7),trunk(argv[2],7,true);std::vector<fs::path> files;for(auto&p:fs::directory_iterator(argv[3]))if(p.path().extension()==".png")files.push_back(p.path());std::sort(files.begin(),files.end());require(files.size()==50,"requires the fixed 50-image validation set");std::ofstream csv(out/"quality.csv");csv<<"image,psnr_db,ssim,max_float_difference,max_gray_difference\n";
 double psnr_sum=0,ssim_sum=0,min_psnr=1e9,min_ssim=1,max_gray=0,max_raw=0;
 for(size_t index=0;index<files.size();++index){auto bgr=cv::imread(files[index].string());cv::Mat rgb;cv::cvtColor(bgr,rgb,cv::COLOR_BGR2RGB);auto a=full.run(rgb),b=trunk.run(rgb);require(a.size()==1&&b.size()==1&&a[0].size()==b[0].size()&&a[0].size()==3*1024*1024,"logical output shape");double mse=0,s=0,raw=0,gray=0;std::vector<cv::Mat> images_a,images_b;
  for(int c=0;c<3;++c){cv::Mat x(1024,1024,CV_32F,a[0].data()+c*1024*1024),y(1024,1024,CV_32F,b[0].data()+c*1024*1024);require(cv::checkRange(x)&&cv::checkRange(y),"nonfinite output");raw=std::max(raw,cv::norm(x,y,cv::NORM_INF));cv::Mat u,v;cv::max(x,0,u);cv::min(u,1,u);cv::max(y,0,v);cv::min(v,1,v);mse+=cv::norm(u,v,cv::NORM_L2SQR)/(3.*1024*1024);s+=ssim(u,v)/3.;cv::Mat xu,yu;u.convertTo(xu,CV_8U,255);v.convertTo(yu,CV_8U,255);gray=std::max(gray,cv::norm(xu,yu,cv::NORM_INF));images_a.push_back(xu);images_b.push_back(yu);}
  double psnr=-10*std::log10(std::max(mse,1e-20));psnr_sum+=psnr;ssim_sum+=s;min_psnr=std::min(min_psnr,psnr);min_ssim=std::min(min_ssim,s);max_gray=std::max(max_gray,gray);max_raw=std::max(max_raw,raw);csv<<files[index].filename().string()<<','<<psnr<<','<<s<<','<<raw<<','<<gray<<'\n';csv.flush();
  if(index==0){cv::Mat image;cv::merge(images_a,image);cv::cvtColor(image,image,cv::COLOR_RGB2BGR);cv::imwrite((out/"reference.png").string(),image);cv::merge(images_b,image);cv::cvtColor(image,image,cv::COLOR_RGB2BGR);cv::imwrite((out/"candidate.png").string(),image);}
 }
 double n=files.size();bool pass=psnr_sum/n>=50&&min_psnr>=40&&ssim_sum/n>=.999&&min_ssim>=.99&&max_gray<=1;cv::FileStorage f((out/"summary.json").string(),cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);f<<"images"<<int(n)<<"mean_psnr_db"<<psnr_sum/n<<"min_psnr_db"<<min_psnr<<"mean_ssim"<<ssim_sum/n<<"min_ssim"<<min_ssim<<"max_float_difference"<<max_raw<<"max_gray_difference"<<max_gray<<"passed"<<int(pass)<<"scope"<<"CPU tail candidate versus validated FP16 full RKNN on the fixed held-out validation set";f.release();std::cout<<"50 image SR candidate parity "<<(pass?"passed":"failed")<<'\n';return pass?0:2;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
