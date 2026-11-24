#include "core.hpp"
#include "rknn_api.h"
#include "sr_tail.hpp"
#include <filesystem>
#include <iostream>
namespace fs=std::filesystem;
using namespace aa;
struct Context {rknn_context value=0;~Context(){if(value)rknn_destroy(value);}};
static void attr(cv::FileStorage& f,const std::string& label,rknn_context ctx,rknn_query_cmd query,unsigned index){
 rknn_tensor_attr a{};a.index=index;int rc=rknn_query(ctx,query,&a,sizeof(a));f<<"{"<<"kind"<<label<<"index"<<int(index)<<"query_rc"<<rc;
 if(!rc){f<<"name"<<a.name<<"type"<<get_type_string(a.type)<<"format"<<get_format_string(a.fmt)<<"n_elems"<<double(a.n_elems)<<"bytes"<<double(a.size)<<"bytes_with_stride"<<double(a.size_with_stride)<<"w_stride"<<int(a.w_stride)<<"dims"<<"[";for(unsigned i=0;i<a.n_dims;++i)f<<int(a.dims[i]);f<<"]";}
 f<<"}";
}
int main(int argc,char**argv){try{
 require(argc==9,"Usage: npu_profile MODEL PREPARED_RGB_IMAGE NEW_OUTPUT_DIR CORE_MASK INIT_FLAGS ITERATIONS standard|prealloc|cpu-tail WARMUPS");
 const std::string mode=argv[7];bool native_tail=mode=="cpu-tail-native";require(mode=="standard"||mode=="prealloc"||mode=="cpu-tail"||native_tail,"output mode");int mask=std::stoi(argv[4]),flags=std::stoi(argv[5]),runs=std::stoi(argv[6]),warmups=std::stoi(argv[8]);require(runs>0&&runs<=100&&warmups>=0&&warmups<=10,"bounded iteration count");
 require(!fs::exists(argv[3]),"output must be new");fs::create_directories(argv[3]);const fs::path out=argv[3];cv::setNumThreads(1);
 std::ifstream mf(argv[1],std::ios::binary);require(bool(mf),"model missing");std::vector<char> bytes((std::istreambuf_iterator<char>(mf)),{});Context context;auto ctx_start=now_us();require(rknn_init(&context.value,bytes.data(),bytes.size(),flags,nullptr)==0,"init");auto ctx=context.value;double init_ms=(now_us()-ctx_start)/1000.;require(rknn_set_core_mask(ctx,static_cast<rknn_core_mask>(mask))==0,"core mask");
 rknn_input_output_num count{};require(rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&count,sizeof(count))==0&&count.n_input==1,"input contract");rknn_tensor_attr input{};require(rknn_query(ctx,RKNN_QUERY_INPUT_ATTR,&input,sizeof(input))==0,"input attr");
 int h=input.fmt==RKNN_TENSOR_NHWC?input.dims[1]:input.dims[2],w=input.fmt==RKNN_TENSOR_NHWC?input.dims[2]:input.dims[3];auto bgr=cv::imread(argv[2]);require(!bgr.empty()&&bgr.cols==w&&bgr.rows==h,"input must already be prepared at model size");cv::Mat rgb;cv::cvtColor(bgr,rgb,cv::COLOR_BGR2RGB);
 std::vector<rknn_tensor_attr> attrs(count.n_output);for(unsigned i=0;i<count.n_output;++i){attrs[i].index=i;require(rknn_query(ctx,RKNN_QUERY_OUTPUT_ATTR,&attrs[i],sizeof(attrs[i]))==0,"output attr");}
 if(mode=="cpu-tail"||native_tail)require(count.n_output==1&&attrs[0].n_dims==4&&attrs[0].dims[0]==1&&attrs[0].dims[1]==48&&attrs[0].dims[2]==256&&attrs[0].dims[3]==256&&attrs[0].type==RKNN_TENSOR_FLOAT16,"FP16 trunk shape");
 std::vector<std::vector<float>> result(count.n_output);std::vector<rknn_output> outputs(count.n_output);std::ofstream csv(out/"timings.csv");csv<<"iteration,warmup,inputs_set_ms,run_ms,allocate_ms,outputs_get_ms,copy_ms,cpu_tail_ms,release_ms,total_ms,driver_run_ms\n";
 for(int iteration=-warmups;iteration<runs;++iteration){
  for(auto& v:result){std::vector<float> empty;v.swap(empty);}auto start=now_us();rknn_input in{};in.type=RKNN_TENSOR_UINT8;in.fmt=RKNN_TENSOR_NHWC;in.size=rgb.total()*3;in.buf=rgb.data;require(rknn_inputs_set(ctx,1,&in)==0,"inputs_set");auto a=now_us();require(rknn_run(ctx,nullptr)==0,"run");auto b=now_us();
  for(unsigned i=0;i<count.n_output;++i){outputs[i]={};outputs[i].index=i;outputs[i].want_float=native_tail?0:1;if(mode=="prealloc"){result[i].resize(attrs[i].n_elems);outputs[i].is_prealloc=1;outputs[i].buf=result[i].data();outputs[i].size=attrs[i].n_elems*sizeof(float);}}
  auto c=now_us();require(rknn_outputs_get(ctx,count.n_output,outputs.data(),nullptr)==0,"outputs_get");auto d=now_us();
  try{if(mode!="prealloc"&&!native_tail)for(unsigned i=0;i<count.n_output;++i){require(outputs[i].size>=attrs[i].n_elems*sizeof(float),"short output");auto ptr=static_cast<float*>(outputs[i].buf);result[i].assign(ptr,ptr+attrs[i].n_elems);}}
  catch(...){rknn_outputs_release(ctx,count.n_output,outputs.data());throw;}
  auto e=now_us();if(native_tail){require(outputs[0].size>=attrs[0].n_elems*sizeof(uint16_t),"short FP16 output");result[0]=sr_cpu_tail_fp16(outputs[0].buf,attrs[0].n_elems,rgb);}else if(mode=="cpu-tail")result[0]=sr_cpu_tail(result[0],rgb);auto tail=now_us();rknn_perf_run perf{};int prc=rknn_query(ctx,RKNN_QUERY_PERF_RUN,&perf,sizeof(perf));require(rknn_outputs_release(ctx,count.n_output,outputs.data())==0,"release");auto end=now_us();
  csv<<iteration<<','<<(iteration<0)<<','<<(a-start)/1000.<<','<<(b-a)/1000.<<','<<(c-b)/1000.<<','<<(d-c)/1000.<<','<<(e-d)/1000.<<','<<(tail-e)/1000.<<','<<(end-tail)/1000.<<','<<(end-start)/1000.<<','<<(prc==0?perf.run_duration/1000.:-1)<<'\n';csv.flush();
 }
 for(unsigned i=0;i<count.n_output;++i){std::ofstream f(out/("output-"+std::to_string(i)+".f32"),std::ios::binary);f.write(reinterpret_cast<const char*>(result[i].data()),result[i].size()*sizeof(float));}
 if(flags&RKNN_FLAG_COLLECT_PERF_MASK){rknn_perf_detail detail{};require(rknn_query(ctx,RKNN_QUERY_PERF_DETAIL,&detail,sizeof(detail))==0&&detail.perf_data,"perf detail query");std::ofstream f(out/"perf-detail.txt",std::ios::binary);f.write(detail.perf_data,detail.data_len);}
 rknn_sdk_version sdk{};require(rknn_query(ctx,RKNN_QUERY_SDK_VERSION,&sdk,sizeof(sdk))==0,"SDK");rknn_mem_size mem{};int mrc=rknn_query(ctx,RKNN_QUERY_MEM_SIZE,&mem,sizeof(mem));
 cv::FileStorage f((out/"metadata.json").string(),cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);f<<"model"<<argv[1]<<"input"<<argv[2]<<"mask"<<mask<<"init_flags"<<flags<<"mode"<<mode<<"runs"<<runs<<"warmups"<<warmups<<"init_ms"<<init_ms<<"sdk"<<sdk.api_version<<"driver"<<sdk.drv_version<<"memory_query_rc"<<mrc;
 if(!mrc)f<<"weight_bytes"<<double(mem.total_weight_size)<<"internal_bytes"<<double(mem.total_internal_size)<<"dma_bytes"<<double(mem.total_dma_allocated_size)<<"total_sram_bytes"<<double(mem.total_sram_size)<<"free_sram_bytes"<<double(mem.free_sram_size);
 f<<"tensors"<<"[";attr(f,"input",ctx,RKNN_QUERY_INPUT_ATTR,0);attr(f,"native_input",ctx,RKNN_QUERY_NATIVE_INPUT_ATTR,0);for(unsigned i=0;i<count.n_output;++i){attr(f,"output",ctx,RKNN_QUERY_OUTPUT_ATTR,i);attr(f,"native_output",ctx,RKNN_QUERY_NATIVE_OUTPUT_ATTR,i);attr(f,"native_nhwc_output",ctx,RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR,i);}f<<"]";f.release();std::cout<<out<<" completed\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
