#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
namespace aa {
inline volatile sig_atomic_t stop_requested=0;
inline int64_t now_us(){return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
inline void require(bool b,const std::string& s){if(!b)throw std::runtime_error(s);}
class Stats {
 std::array<uint64_t,20001> bins{};uint64_t n=0;double maximum=0;
public:
 void add(double ms){++n;maximum=std::max(maximum,ms);++bins[std::min(size_t(20000),size_t(std::ceil(std::max(0.,ms)*10)))];}
 double quantile(double q)const{if(!n)return 0;uint64_t target=std::max(uint64_t(1),uint64_t(std::ceil(q*n))),sum=0;for(size_t i=0;i<bins.size();++i){sum+=bins[i];if(sum>=target)return i==20000?maximum:i/10.;}return maximum;}
 double max()const{return maximum;}
};
enum class CaptureMatrix {BT709,BT601,BT2020};
struct Frame {uint64_t id=0;int64_t captured_us=0;cv::Mat rgb,nv12;CaptureMatrix matrix=CaptureMatrix::BT709;bool full_range=false;};
template<class T> class Latest {
 std::mutex mu;std::condition_variable cv;std::shared_ptr<T> item;bool closed=false;uint64_t highest=0;
public:
 std::atomic<uint64_t> replaced{0},out_of_order{0};
 void put(std::shared_ptr<T> p){std::lock_guard<std::mutex> l(mu);if(closed)return;if(p->id<=highest){++out_of_order;return;}highest=p->id;if(item)++replaced;item=std::move(p);cv.notify_one();}
 std::shared_ptr<T> get(){std::unique_lock<std::mutex> l(mu);cv.wait_for(l,std::chrono::milliseconds(200),[&]{return item||closed;});auto p=std::move(item);return p;}
 void close(){std::lock_guard<std::mutex> l(mu);closed=true;item.reset();cv.notify_all();}
};
struct Letterbox {cv::Mat rgb;double sx=1,sy=1;int x=0,y=0,w=0,h=0;};
cv::Mat cpu_nv12_rgb(const cv::Mat&,int,int,double,double,bool);
class RgaResize {struct Impl;std::unique_ptr<Impl> p;public:RgaResize();~RgaResize();cv::Mat resize(const cv::Mat&,int,int);cv::Mat nv12_rgb(const cv::Mat&,int,int);Letterbox letterbox(const cv::Mat&,int);Letterbox letterbox(const cv::Mat&,int,int);};
void materialize_rgb(Frame&,RgaResize&);
class Model {struct Impl;std::unique_ptr<Impl> p;public:int width=0,height=0;std::string sdk;std::vector<std::vector<uint32_t>> shapes;Model(const std::string&,int,bool sr_tail=false);~Model();std::vector<std::vector<float>> run(const cv::Mat&);};
class Camera {struct Impl;std::unique_ptr<Impl> p;public:Camera(int,int,const std::string&);~Camera();std::shared_ptr<Frame> take(bool defer_rgb=false);};
class Display {struct Impl;std::unique_ptr<Impl> p;public:Display();~Display();int64_t show(const cv::Mat&);};
struct Detection {cv::Rect2f box;float score;int label;};
std::vector<Detection> decode_yolo(const std::vector<std::vector<float>>&,const Letterbox&,int,int,float);
struct Counts {uint64_t forward=0,backward=0;std::map<int,int> side;std::map<int,int64_t> last;
 int update(int id,double position,int64_t time_us,double line=.5,double band=.015){
  if(position>=line-band&&position<=line+band)return 0;
  int s=position<line?-1:1,event=0;
  if(side.count(id)&&side[id]!=s){event=s>0?1:-1;if(event>0)++forward;else ++backward;}
  side[id]=s;last[id]=time_us;return event;
 }
 void expire(int64_t now){for(auto i=last.begin();i!=last.end();)if(now-i->second>10000000){side.erase(i->first);i=last.erase(i);}else ++i;}
};
}
