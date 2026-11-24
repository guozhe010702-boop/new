#include "core.hpp"
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>
namespace aa {
struct Display::Impl {
 int fd=-1;uint32_t connector=0,crtc=0;drmModeModeInfo mode{};int index=0;cv::Mat scaled;bool active=false,pending=false;int64_t flip_us=0;
 struct Buffer{int content_w=0,content_h=0;uint32_t handle=0,fb=0,pitch=0;uint64_t size=0;void* map=MAP_FAILED;} buffers[2];
 static void flipped(int,unsigned,unsigned sec,unsigned usec,void* p){auto self=static_cast<Impl*>(p);self->flip_us=int64_t(sec)*1000000+usec;self->pending=false;}
 ~Impl(){if(fd<0)return;if(active)drmModeSetCrtc(fd,crtc,0,0,0,nullptr,0,nullptr);for(auto& b:buffers){if(b.fb)drmModeRmFB(fd,b.fb);if(b.map!=MAP_FAILED)munmap(b.map,b.size);if(b.handle){drm_mode_destroy_dumb d{};d.handle=b.handle;ioctl(fd,DRM_IOCTL_MODE_DESTROY_DUMB,&d);}}drmDropMaster(fd);close(fd);}
};
Display::Display():p(new Impl){
 p->fd=open("/dev/dri/card0",O_RDWR|O_CLOEXEC);require(p->fd>=0,"open DRM");require(drmSetMaster(p->fd)==0,"DRM master unavailable; stop desktop through control.py");
 uint64_t mono=0;require(drmGetCap(p->fd,DRM_CAP_TIMESTAMP_MONOTONIC,&mono)==0&&mono,"DRM monotonic flip timestamps required");
 auto res=drmModeGetResources(p->fd);require(res,"DRM resources");
 for(int i=0;i<res->count_connectors&&!p->connector;++i){auto c=drmModeGetConnector(p->fd,res->connectors[i]);if(c&&c->connection==DRM_MODE_CONNECTED&&c->connector_type==DRM_MODE_CONNECTOR_HDMIA){
  for(int j=0;j<c->count_modes;++j){auto m=c->modes[j];double hz=double(m.clock)*1000/m.htotal/m.vtotal;if(m.hdisplay==1920&&m.vdisplay==1080&&hz>59&&hz<61){p->connector=c->connector_id;p->mode=m;break;}}
  if(p->connector){for(int j=0;j<c->count_encoders&&!p->crtc;++j){auto enc=drmModeGetEncoder(p->fd,c->encoders[j]);if(enc){if(enc->crtc_id)p->crtc=enc->crtc_id;else for(int k=0;k<res->count_crtcs;++k)if(enc->possible_crtcs&(1<<k)){p->crtc=res->crtcs[k];break;}drmModeFreeEncoder(enc);}}}
 }if(c)drmModeFreeConnector(c);}
 drmModeFreeResources(res);require(p->connector&&p->crtc,"HDMI 1080p60 unavailable");
 for(auto& b:p->buffers){drm_mode_create_dumb d{};d.width=1920;d.height=1080;d.bpp=32;require(ioctl(p->fd,DRM_IOCTL_MODE_CREATE_DUMB,&d)==0,"DRM dumb allocate");b.handle=d.handle;b.pitch=d.pitch;b.size=d.size;require(drmModeAddFB(p->fd,1920,1080,24,32,b.pitch,b.handle,&b.fb)==0,"DRM framebuffer");drm_mode_map_dumb map{};map.handle=b.handle;require(ioctl(p->fd,DRM_IOCTL_MODE_MAP_DUMB,&map)==0,"DRM map offset");b.map=mmap(nullptr,b.size,PROT_READ|PROT_WRITE,MAP_SHARED,p->fd,map.offset);require(b.map!=MAP_FAILED,"DRM mmap");memset(b.map,0,b.size);}
}
Display::~Display()=default;
int64_t Display::show(const cv::Mat& rgb){
 auto& b=p->buffers[p->index];cv::Mat canvas(1080,1920,CV_8UC4,b.map,b.pitch),fitted;double scale=std::min(1920.0/rgb.cols,1080.0/rgb.rows);int w=std::max(1,int(std::round(rgb.cols*scale))),h=std::max(1,int(std::round(rgb.rows*scale)));if(b.content_w!=w||b.content_h!=h){canvas.setTo(cv::Scalar(0,0,0,255));b.content_w=w;b.content_h=h;}if(rgb.cols==w&&rgb.rows==h)fitted=rgb;else{cv::resize(rgb,p->scaled,{w,h});fitted=p->scaled;}auto target=canvas(cv::Rect((1920-w)/2,(1080-h)/2,w,h));cv::cvtColor(fitted,target,cv::COLOR_RGB2BGRA);
 if(!p->active){require(drmModeSetCrtc(p->fd,p->crtc,b.fb,0,0,&p->connector,1,&p->mode)==0,"HDMI set mode");p->active=true;p->flip_us=now_us();}
 else {p->pending=true;require(drmModePageFlip(p->fd,p->crtc,b.fb,DRM_MODE_PAGE_FLIP_EVENT,p.get())==0,"HDMI page flip");int64_t deadline=now_us()+1500000;while(p->pending){require(now_us()<deadline,"HDMI flip timeout");pollfd f{p->fd,POLLIN,0};int rc=poll(&f,1,100);if(rc<0&&errno==EINTR)continue;require(rc>=0,"DRM poll");if(rc>0){drmEventContext ev{};ev.version=2;ev.page_flip_handler=Impl::flipped;require(drmHandleEvent(p->fd,&ev)==0,"DRM event");}}}
 p->index=1-p->index;return p->flip_us;
}
}
