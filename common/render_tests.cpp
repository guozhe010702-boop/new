#include "render.hpp"
#include <iostream>
#include <random>
using namespace aa;
int main(){try{
 cv::setNumThreads(1);std::mt19937 rng(123);std::uniform_real_distribution<float> dist(-.2,1.2);std::vector<float> v(1024*1024*3);for(auto& x:v)x=dist(rng);
 auto reference=rgb_from_planar(v,1024,1024,false);auto actual=rgb_from_planar(v,1024,1024,true);double difference=cv::norm(reference,actual,cv::NORM_INF);require(difference<=1,"vector renderer exceeds one gray-level rounding tolerance");
 double scalar=0,vector=0;for(int i=0;i<20;++i){auto t=now_us();auto a=rgb_from_planar(v,1024,1024,false);scalar+=(now_us()-t)/1000.;t=now_us();auto b=rgb_from_planar(v,1024,1024,true);vector+=(now_us()-t)/1000.;require(a.size()==b.size(),"geometry");}
 std::cout<<"{\"max_pixel_difference\":"<<difference<<",\"scalar_ms\":"<<scalar/20<<",\"vector_ms\":"<<vector/20<<",\"trials\":20}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
