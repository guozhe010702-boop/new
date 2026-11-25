#include "core.hpp"
#include "im2d.h"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
namespace aa {
cv::Mat cpu_nv12_rgb(const cv::Mat& nv,int w,int h,double kr,double kb,bool full){
 cv::Mat uv(h/2,w/2,CV_8UC2,const_cast<uint8_t*>(nv.ptr(h)));cv::Mat expanded;cv::resize(uv,expanded,{w,h},0,0,cv::INTER_NEAREST);std::vector<cv::Mat> channels;cv::split(expanded,channels);cv::Mat packed,rgb;cv::merge(std::vector<cv::Mat>{nv.rowRange(0,h),channels[0],channels[1]},packed);
 double kg=1-kr-kb,ys=full?1.:255./219.,cs=full?1.:255./224.,rv=2*(1-kr)*cs,bu=2*(1-kb)*cs,gu=-2*kb*(1-kb)/kg*cs,gv=-2*kr*(1-kr)/kg*cs,ybias=full?0:-16*ys;
 cv::Mat matrix=(cv::Mat_<float>(3,4)<<ys,0,rv,ybias-128*rv,ys,gu,gv,ybias-128*(gu+gv),ys,bu,0,ybias-128*bu);cv::transform(packed,rgb,matrix);return rgb;
}
struct Dma {
 struct Allocation{uint64_t len;uint32_t fd,flags;uint64_t heap_flags;};
 int fd=-1;void* data=MAP_FAILED;size_t size;
 Dma(size_t n):size(n){int heap=open("/dev/dma_heap/system-dma32",O_RDWR|O_CLOEXEC);require(heap>=0,"DMA32 heap missing");Allocation a{(n+4095)&~size_t(4095),0,O_RDWR|O_CLOEXEC,0};int r=ioctl(heap,_IOWR('H',0,Allocation),&a);close(heap);require(r==0,"DMA32 allocate failed");fd=a.fd;data=mmap(nullptr,size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);if(data==MAP_FAILED){close(fd);fd=-1;throw std::runtime_error("mmap DMA failed");}}
 ~Dma(){if(data!=MAP_FAILED)munmap(data,size);if(fd>=0)close(fd);}
 void sync(bool end,bool write){struct {uint64_t flags;} s{uint64_t((end?4:0)|(write?2:1))};require(ioctl(fd,_IOW('b',0,decltype(s)),&s)==0,"DMA cache sync");}
};
struct RgaResize::Impl{std::unique_ptr<Dma> src,dst,nv12,rgb;int sw=0,sh=0,dw=0,dh=0,cw=0,ch=0;};
RgaResize::RgaResize():p(new Impl){}RgaResize::~RgaResize()=default;
void materialize_rgb(Frame& frame,RgaResize& rga){
 if(!frame.rgb.empty())return;
 require(frame.nv12.type()==CV_8UC1&&frame.nv12.rows%3==0&&frame.nv12.cols%2==0,"invalid packed NV12 frame");
 int w=frame.nv12.cols,h=frame.nv12.rows*2/3;bool full=frame.full_range;
 if(frame.matrix==CaptureMatrix::BT709&&full)frame.rgb=rga.nv12_rgb(frame.nv12,w,h);
 else if(frame.matrix==CaptureMatrix::BT2020)frame.rgb=cpu_nv12_rgb(frame.nv12,w,h,.2627,.0593,full);
 else if(frame.matrix==CaptureMatrix::BT601)frame.rgb=cpu_nv12_rgb(frame.nv12,w,h,.299,.114,full);
 else if(frame.matrix==CaptureMatrix::BT709)frame.rgb=cpu_nv12_rgb(frame.nv12,w,h,.2126,.0722,full);
 else throw std::runtime_error("Unsupported capture color matrix");
 frame.nv12.release();
}
cv::Mat RgaResize::nv12_rgb(const cv::Mat& image,int w,int h){
 if(p->cw!=w||p->ch!=h){p->nv12.reset(new Dma(size_t(w)*h*3/2));p->rgb.reset(new Dma(size_t(w)*h*3));p->cw=w;p->ch=h;}
 p->nv12->sync(false,true);auto dst=static_cast<uint8_t*>(p->nv12->data);auto src=image.data;size_t pixels=size_t(w)*h;
 // Known BSP full-range -> limited remap; RGA BT709-limited CSC is verified
 // on this board, whereas the full-range hardware mode was not reliable.
 for(size_t i=0;i<pixels;++i)dst[i]=16+(unsigned(src[i])*219+127)/255;
 for(size_t i=pixels;i<pixels*3/2;++i)dst[i]=16+(unsigned(src[i])*224+127)/255;
 p->nv12->sync(true,true);
 auto a=wrapbuffer_fd_t(p->nv12->fd,w,h,w,h,RK_FORMAT_YCbCr_420_SP),b=wrapbuffer_fd_t(p->rgb->fd,w,h,w,h,RK_FORMAT_RGB_888);
 auto status=imcvtcolor(a,b,RK_FORMAT_YCbCr_420_SP,RK_FORMAT_RGB_888,IM_YUV_TO_RGB_BT709_LIMIT);require(status==IM_STATUS_SUCCESS,std::string("RGA CSC: ")+imStrError_t(status));
 p->rgb->sync(false,false);auto result=cv::Mat(h,w,CV_8UC3,p->rgb->data).clone();p->rgb->sync(true,false);return result;
}
cv::Mat RgaResize::resize(const cv::Mat& image,int w,int h){
 require(image.type()==CV_8UC3,"RGA input must be RGB8");
 int ss=(image.cols+15)/16*16,ds=(w+15)/16*16;
 if(p->sw!=image.cols||p->sh!=image.rows){p->src.reset(new Dma(size_t(ss)*image.rows*3));p->sw=image.cols;p->sh=image.rows;}
 if(p->dw!=w||p->dh!=h){p->dst.reset(new Dma(size_t(ds)*h*3));p->dw=w;p->dh=h;}
 p->src->sync(false,true);for(int y=0;y<image.rows;++y)memcpy(static_cast<uint8_t*>(p->src->data)+size_t(y)*ss*3,image.ptr(y),image.cols*3);p->src->sync(true,true);
 auto src=wrapbuffer_fd_t(p->src->fd,image.cols,image.rows,ss,image.rows,RK_FORMAT_RGB_888);auto dst=wrapbuffer_fd_t(p->dst->fd,w,h,ds,h,RK_FORMAT_RGB_888);
 auto status=imresize(src,dst);require(status==IM_STATUS_SUCCESS,std::string("RGA resize: ")+imStrError_t(status));
 p->dst->sync(false,false);cv::Mat output=cv::Mat(h,w,CV_8UC3,p->dst->data,ds*3).clone();p->dst->sync(true,false);return output;
}
Letterbox RgaResize::letterbox(const cv::Mat& im,int size){return letterbox(im,size,size);}
Letterbox RgaResize::letterbox(const cv::Mat& im,int width,int height){double scale=std::min(double(width)/im.cols,double(height)/im.rows);int w=std::max(1,int(std::round(im.cols*scale))),h=std::max(1,int(std::round(im.rows*scale)));Letterbox b;b.w=w;b.h=h;b.x=(width-w)/2;b.y=(height-h)/2;b.sx=double(w)/im.cols;b.sy=double(h)/im.rows;b.rgb=cv::Mat(height,width,CV_8UC3,cv::Scalar(114,114,114));resize(im,w,h).copyTo(b.rgb(cv::Rect(b.x,b.y,w,h)));return b;}
}
