#include "core.hpp"
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <iostream>
namespace aa {
struct Camera::Impl{GstElement* pipe=nullptr;GstAppSink* sink=nullptr;uint64_t seq=0;RgaResize color;~Impl(){if(pipe){gst_element_set_state(pipe,GST_STATE_NULL);gst_object_unref(pipe);}if(sink)gst_object_unref(sink);}};
Camera::Camera(int w,int h,const std::string& memory):p(new Impl){
 gst_init(nullptr,nullptr);std::string pipeline="v4l2src device=/dev/video44 io-mode="+memory+" ! video/x-raw,format=NV12,width="+std::to_string(w)+",height="+std::to_string(h)+" ! videorate drop-only=true skip-to-first=true ! video/x-raw,framerate=30/1 ! appsink name=frames max-buffers=1 drop=true sync=false";
 GError* error=nullptr;p->pipe=gst_parse_launch(pipeline.c_str(),&error);if(error){std::string s=error->message;g_error_free(error);throw std::runtime_error(s);}require(p->pipe,"GStreamer pipeline missing");p->sink=GST_APP_SINK(gst_bin_get_by_name(GST_BIN(p->pipe),"frames"));require(p->sink,"appsink missing");require(gst_element_set_state(p->pipe,GST_STATE_PLAYING)!=GST_STATE_CHANGE_FAILURE,"camera GStreamer start failed");
}
Camera::~Camera()=default;
std::shared_ptr<Frame> Camera::take(bool defer_rgb){
 GstSample* sample=gst_app_sink_try_pull_sample(p->sink,1000*GST_MSECOND);
 if(!sample){auto bus=gst_element_get_bus(p->pipe);auto msg=gst_bus_pop_filtered(bus,GST_MESSAGE_ERROR);gst_object_unref(bus);if(msg){GError* err=nullptr;gchar* debug=nullptr;gst_message_parse_error(msg,&err,&debug);std::string s=err?err->message:"GStreamer camera error";g_clear_error(&err);g_free(debug);gst_message_unref(msg);throw std::runtime_error(s);}return {};}
 GstVideoInfo info{};GstVideoFrame mapped{};bool valid=false;
 try{
  require(gst_video_info_from_caps(&info,gst_sample_get_caps(sample)),"invalid camera caps");require(GST_VIDEO_INFO_FORMAT(&info)==GST_VIDEO_FORMAT_NV12,"camera must negotiate NV12");
  auto f=std::make_shared<Frame>();f->id=++p->seq;f->captured_us=now_us();
  valid=gst_video_frame_map(&mapped,&info,gst_sample_get_buffer(sample),GST_MAP_READ);require(valid,"camera map failed");int w=GST_VIDEO_INFO_WIDTH(&info),h=GST_VIDEO_INFO_HEIGHT(&info);cv::Mat nv12(h*3/2,w,CV_8UC1);
  require(GST_VIDEO_FRAME_PLANE_STRIDE(&mapped,0)>=w&&GST_VIDEO_FRAME_PLANE_STRIDE(&mapped,1)>=w,"camera stride too small");
  for(int y=0;y<h;++y)memcpy(nv12.ptr(y),static_cast<uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&mapped,0))+y*GST_VIDEO_FRAME_PLANE_STRIDE(&mapped,0),w);
  for(int y=0;y<h/2;++y)memcpy(nv12.ptr(h+y),static_cast<uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&mapped,1))+y*GST_VIDEO_FRAME_PLANE_STRIDE(&mapped,1),w);
  gst_video_frame_unmap(&mapped);valid=false;gst_sample_unref(sample);sample=nullptr;
  bool full=info.colorimetry.range==GST_VIDEO_COLOR_RANGE_0_255;
  if(f->id==1){gchar* desc=gst_video_colorimetry_to_string(&info.colorimetry);std::cerr<<"CAPTURE colorimetry="<<(desc?desc:"unknown")<<" actual="<<w<<'x'<<h<<" CSC="<<(info.colorimetry.matrix==GST_VIDEO_COLOR_MATRIX_BT709&&full?"RGA709 + CPU range remap":"CPU matrix; RGA model resize")<<'\n';g_free(desc);}
  if(info.colorimetry.matrix==GST_VIDEO_COLOR_MATRIX_BT709)f->matrix=CaptureMatrix::BT709;
  else if(info.colorimetry.matrix==GST_VIDEO_COLOR_MATRIX_BT2020)f->matrix=CaptureMatrix::BT2020;
  else if(info.colorimetry.matrix==GST_VIDEO_COLOR_MATRIX_BT601)f->matrix=CaptureMatrix::BT601;
  else throw std::runtime_error("Camera matrix not supported; refusing guessed colors");
  f->full_range=full;f->nv12=std::move(nv12);if(!defer_rgb)materialize_rgb(*f,p->color);
  return f;
 }catch(...){if(valid)gst_video_frame_unmap(&mapped);if(sample)gst_sample_unref(sample);throw;}
}
}
