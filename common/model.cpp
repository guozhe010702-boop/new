#include "core.hpp"
#include "rknn_api.h"
#include "sr_tail.hpp"
#include <numeric>
namespace aa {
struct Model::Impl {
 rknn_context ctx=0;rknn_input_output_num count{};std::vector<rknn_tensor_attr> attrs;bool sr_tail=false;
 ~Impl(){if(ctx)rknn_destroy(ctx);}
};
Model::Model(const std::string& path,int mask,bool sr_tail):p(new Impl){
 p->sr_tail=sr_tail;
 std::ifstream f(path,std::ios::binary);require(bool(f),"model missing: "+path);std::vector<char> bytes((std::istreambuf_iterator<char>(f)),{});
 require(rknn_init(&p->ctx,bytes.data(),bytes.size(),0,nullptr)==0,"rknn_init failed");
 require(rknn_set_core_mask(p->ctx,static_cast<rknn_core_mask>(mask))==0,"RKNN core mask rejected");
 rknn_sdk_version v{};require(rknn_query(p->ctx,RKNN_QUERY_SDK_VERSION,&v,sizeof(v))==0,"query SDK");sdk=std::string(v.api_version)+" driver="+v.drv_version;
 require(rknn_query(p->ctx,RKNN_QUERY_IN_OUT_NUM,&p->count,sizeof(p->count))==0&&p->count.n_input==1,"expected one image input");
 rknn_tensor_attr in{};require(rknn_query(p->ctx,RKNN_QUERY_INPUT_ATTR,&in,sizeof(in))==0&&in.n_dims==4,"input attrs");
 if(in.fmt==RKNN_TENSOR_NHWC){height=in.dims[1];width=in.dims[2];require(in.dims[3]==3,"input channels");}
 else {height=in.dims[2];width=in.dims[3];require(in.dims[1]==3,"input channels");}
 for(unsigned i=0;i<p->count.n_output;++i){rknn_tensor_attr a{};a.index=i;require(rknn_query(p->ctx,RKNN_QUERY_OUTPUT_ATTR,&a,sizeof(a))==0,"output attrs");p->attrs.push_back(a);shapes.emplace_back(a.dims,a.dims+a.n_dims);}
 if(sr_tail){require(width==256&&height==256&&shapes==std::vector<std::vector<uint32_t>>{{1,48,256,256}}&&p->attrs[0].type==RKNN_TENSOR_FLOAT16,"SR CPU tail requires unchanged 256 input / FP16 48-channel NCHW trunk");shapes={{1,3,1024,1024}};}
}
Model::~Model()=default;
std::vector<std::vector<float>> Model::run(const cv::Mat& rgb){
 require(rgb.type()==CV_8UC3&&rgb.cols==width&&rgb.rows==height&&rgb.isContinuous(),"model expects contiguous uint8 RGB exact size");
 rknn_input in{};in.index=0;in.type=RKNN_TENSOR_UINT8;in.fmt=RKNN_TENSOR_NHWC;in.size=rgb.total()*3;in.buf=rgb.data;in.pass_through=0;
 require(rknn_inputs_set(p->ctx,1,&in)==0,"RKNN inputs_set");require(rknn_run(p->ctx,nullptr)==0,"RKNN run");
 if(p->sr_tail){
  rknn_output output{};output.index=0;output.want_float=0;require(rknn_outputs_get(p->ctx,1,&output,nullptr)==0,"RKNN FP16 output");std::vector<float> reconstructed;
  try{require(output.size>=p->attrs[0].n_elems*sizeof(uint16_t),"short FP16 output");reconstructed=sr_cpu_tail_fp16(output.buf,p->attrs[0].n_elems,rgb);}catch(...){rknn_outputs_release(p->ctx,1,&output);throw;}
  require(rknn_outputs_release(p->ctx,1,&output)==0,"RKNN release FP16 output");std::vector<std::vector<float>> result;result.push_back(std::move(reconstructed));return result;
 }
 std::vector<rknn_output> outputs(p->count.n_output);for(unsigned i=0;i<outputs.size();++i){outputs[i].index=i;outputs[i].want_float=1;}
 require(rknn_outputs_get(p->ctx,outputs.size(),outputs.data(),nullptr)==0,"RKNN outputs_get");
 std::vector<std::vector<float>> result;
 try{for(unsigned i=0;i<outputs.size();++i){require(outputs[i].size>=p->attrs[i].n_elems*sizeof(float),"short output");auto a=static_cast<float*>(outputs[i].buf);result.emplace_back(a,a+p->attrs[i].n_elems);}}
 catch(...){rknn_outputs_release(p->ctx,outputs.size(),outputs.data());throw;}
 require(rknn_outputs_release(p->ctx,outputs.size(),outputs.data())==0,"RKNN release");return result;
}
std::vector<Detection> decode_yolo(const std::vector<std::vector<float>>& o,const Letterbox& lb,int w,int h,float threshold){
 require(o.size()==9,"YOLO11 requires 9-output Rockchip DFL graph");std::vector<Detection> candidates;
 for(int branch=0;branch<3;++branch){int grid=80>>branch,n=grid*grid;auto& box=o[branch*3];auto& cls=o[branch*3+1];require(box.size()==size_t(n*64)&&cls.size()==size_t(n*80),"YOLO tensor contract mismatch");
  for(int k=0;k<n;++k){int label=0;float score=cls[k];for(int c=1;c<80;++c)if(cls[c*n+k]>score){label=c;score=cls[c*n+k];}if(score<threshold)continue;
   float dist[4]{};for(int d=0;d<4;++d){float mx=-1e9,den=0;for(int j=0;j<16;++j)mx=std::max(mx,box[(d*16+j)*n+k]);for(int j=0;j<16;++j){float e=std::exp(box[(d*16+j)*n+k]-mx);dist[d]+=j*e;den+=e;}dist[d]/=den;}
   float stride=640.f/grid,x=(k%grid+.5f)*stride,y=(k/grid+.5f)*stride;
   float x1=std::clamp(float((x-dist[0]*stride-lb.x)/lb.sx),0.f,float(w)),y1=std::clamp(float((y-dist[1]*stride-lb.y)/lb.sy),0.f,float(h));
   float x2=std::clamp(float((x+dist[2]*stride-lb.x)/lb.sx),0.f,float(w)),y2=std::clamp(float((y+dist[3]*stride-lb.y)/lb.sy),0.f,float(h));
   if(x2>x1&&y2>y1)candidates.push_back({{x1,y1,x2-x1,y2-y1},score,label});
  }
 }
 std::sort(candidates.begin(),candidates.end(),[](auto&a,auto&b){return a.score>b.score;});std::vector<Detection> keep;
 for(auto& d:candidates){bool suppressed=false;for(auto& k:keep)if(k.label==d.label){double inter=(k.box&d.box).area(),u=k.box.area()+d.box.area()-inter;if(u>0&&inter/u>.45){suppressed=true;break;}}if(!suppressed)keep.push_back(d);if(keep.size()>=300)break;}return keep;
}
}
