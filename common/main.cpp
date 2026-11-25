#include "core.hpp"
#include "render.hpp"
#include "async_writer.hpp"
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include "BYTETracker.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>
namespace fs=std::filesystem;
using namespace aa;
struct Options {
 std::string project,model,out,image,video,memory="mmap",render="vector",sr_tail="npu",capture_convert="eager",depth_profile="standard",inference_cpus="auto";int seconds=60,workers=1,mask=7,save_every=300,stall_ms=0;int width=1920,height=1080,roi_x=-1,roi_y=-1;int display_error_after=0;bool hdmi=false;double line=.5;int target_class=0;
};
Options parse(int argc,char**argv){Options o;for(int i=1;i<argc;++i){std::string k=argv[i];require(i+1<argc,"missing option value");std::string v=argv[++i];if(k=="--project")o.project=v;else if(k=="--render")o.render=v;else if(k=="--sr-tail")o.sr_tail=v;else if(k=="--model")o.model=v;else if(k=="--output")o.out=v;else if(k=="--image")o.image=v;else if(k=="--video")o.video=v;else if(k=="--depth-profile")o.depth_profile=v;else if(k=="--inference-cpus")o.inference_cpus=v;else if(k=="--capture-convert")o.capture_convert=v;else if(k=="--capture-memory")o.memory=v;else if(k=="--seconds")o.seconds=std::stoi(v);else if(k=="--workers")o.workers=std::stoi(v);else if(k=="--mask")o.mask=std::stoi(v);else if(k=="--save-every")o.save_every=std::stoi(v);else if(k=="--inject-display-error-after")o.display_error_after=std::stoi(v);else if(k=="--hdmi")o.hdmi=std::stoi(v);else if(k=="--width")o.width=std::stoi(v);else if(k=="--height")o.height=std::stoi(v);else if(k=="--roi-x")o.roi_x=std::stoi(v);else if(k=="--roi-y")o.roi_y=std::stoi(v);else if(k=="--line")o.line=std::stod(v);else if(k=="--class")o.target_class=std::stoi(v);else if(k=="--inject-stall-ms")o.stall_ms=std::stoi(v);else throw std::runtime_error("unknown argument "+k);}
 require(!o.out.empty()&&(o.project=="yolo11_bytetrack"||o.project=="depth_anything_v2"||o.project=="real_esrgan"||o.project=="capture"),"project/output required");require(o.workers>=1&&o.workers<=3&&o.mask>=1&&o.mask<=7&&o.seconds>=0&&o.save_every>=0&&o.line>0&&o.line<1&&o.target_class>=-1&&o.target_class<80&&o.width>=256&&o.height>=256&&o.stall_ms>=0&&o.stall_ms<=2000,"invalid parameters");require(o.render=="scalar"||o.render=="vector","render must be scalar/vector");require(o.sr_tail=="npu"||(o.sr_tail=="cpu"&&o.project=="real_esrgan"),"sr-tail cpu requires real_esrgan");require(o.inference_cpus=="auto"||o.inference_cpus=="big","inference-cpus must be auto or big");require(o.depth_profile=="standard"||(o.depth_profile=="camera-rect"&&o.project=="depth_anything_v2"),"camera-rect requires depth project");require(o.capture_convert=="eager"||o.capture_convert=="selected","capture-convert must be eager or selected");require(o.memory=="mmap"||o.memory=="dmabuf","capture-memory must be mmap or dmabuf");require(o.image.empty()||o.video.empty(),"choose image or video");return o;}
void save_rgb(const fs::path& p,const cv::Mat& rgb){cv::Mat bgr;cv::cvtColor(rgb,bgr,cv::COLOR_RGB2BGR);require(cv::imwrite(p.string(),bgr),"save image");}
std::string metrics(){
 std::ostringstream s;double cpu=-1;static uint64_t prev_ticks=0;static int64_t prev_time=0;std::ifstream stat("/proc/self/stat");std::string raw;std::getline(stat,raw);if(raw.rfind(')')!=std::string::npos){std::istringstream fields(raw.substr(raw.rfind(')')+2));std::vector<std::string> v;std::string word;while(fields>>word)v.push_back(word);if(v.size()>12){auto ticks=std::stoull(v[11])+std::stoull(v[12]);auto t=now_us();if(prev_time)cpu=100.*(ticks-prev_ticks)/sysconf(_SC_CLK_TCK)/((t-prev_time)/1000000.);prev_ticks=ticks;prev_time=t;}}double temp=-1;for(auto& p:fs::directory_iterator("/sys/class/thermal"))if(p.path().filename().string().find("thermal_zone")==0){std::ifstream f(p.path()/"temp");double t;if(f>>t)temp=std::max(temp,t/1000);}
 std::ifstream mem("/proc/self/status");std::string line;int rss=0;while(std::getline(mem,line))if(line.rfind("VmRSS:",0)==0){std::istringstream v(line.substr(6));v>>rss;}
 std::ifstream load("/sys/kernel/debug/rknpu/load");std::string npu;std::getline(load,npu);s<<"temp="<<std::fixed<<std::setprecision(1)<<temp<<"C CPU="<<cpu<<"% RSS="<<rss/1024<<"MiB "<<npu;return s.str();
}
struct Result{uint64_t id=0;std::shared_ptr<Frame> frame;cv::Mat prepared;Letterbox lb;std::vector<std::vector<float>> output;double prep_ms=0,infer_ms=0;};
struct SaveJob {std::shared_ptr<Result> result;cv::Mat relative_depth,sr,preview;uint64_t index=0;};
int main(int argc,char**argv){
 try{
  auto o=parse(argc,argv);require(!fs::exists(o.out),"output must be a new directory");fs::create_directories(o.out);cv::setNumThreads(1);
  std::signal(SIGINT,[](int){stop_requested=1;});std::signal(SIGTERM,[](int){stop_requested=1;});
  std::vector<std::unique_ptr<Model>> models;std::vector<std::unique_ptr<RgaResize>> resize;
  if(o.project!="capture")for(int i=0;i<o.workers;++i){models.emplace_back(new Model(o.model,o.workers==1?o.mask:1<<i,o.sr_tail=="cpu"));resize.emplace_back(new RgaResize);}
  if(o.depth_profile=="camera-rect")require(models[0]->width==518&&models[0]->height==294,"camera-rect requires its 518x294 model");
  std::ofstream meta(fs::path(o.out)/"runtime.txt");meta<<"project="<<o.project<<"\nworkers="<<o.workers<<"\nmask="<<o.mask<<"\ndepth_profile="<<o.depth_profile<<"\ninference_cpus="<<o.inference_cpus<<"\ncapture_convert="<<o.capture_convert<<"\nsave_every="<<o.save_every<<"\nsave_policy=bounded_async_latest"<<"\nsr_tail="<<o.sr_tail<<"\ncapture_memory="<<o.memory<<"\ncapture_adapter=GStreamer NV12 map/copy + CSC selected from caps: RGA BT709-full with CPU range remap; CPU matrix for BT2020/limited; RGA RGB resize\n";if(!models.empty()){meta<<"sdk="<<models[0]->sdk<<"\ninput="<<models[0]->width<<"x"<<models[0]->height<<"\n";for(auto& s:models[0]->shapes){for(auto v:s)meta<<v<<',';meta<<'\n';}}meta.close();
  std::unique_ptr<Display> display;if(o.hdmi)display.reset(new Display);
  Latest<Frame> queue;Latest<Result> results;std::atomic<bool> running{true},capture_done{false};std::atomic<uint64_t> captured{0},inferred{0};std::mutex error_mu;std::string error;
  auto fail=[&](const std::exception& e){std::lock_guard<std::mutex> l(error_mu);if(error.empty())error=e.what();running=false;queue.close();results.close();};
  AsyncLatestWriter<SaveJob> saver([&](const std::shared_ptr<SaveJob>& job){
   setpriority(PRIO_PROCESS,static_cast<id_t>(syscall(SYS_gettid)),10);auto& r=job->result;auto& f=r->frame;auto folder=fs::path(o.out);
   if(!job->relative_depth.empty()){cv::FileStorage store((folder/("depth-"+std::to_string(f->id)+".yml.gz")).string(),cv::FileStorage::WRITE);require(store.isOpened(),"open depth output");store<<"relative_depth"<<job->relative_depth;}
   if(!job->sr.empty())save_rgb(folder/("sr-"+std::to_string(f->id)+".png"),job->sr);
   save_rgb(folder/("input-"+std::to_string(f->id)+".png"),f->rgb);save_rgb(folder/("prepared-"+std::to_string(f->id)+".png"),r->prepared);
   for(size_t j=0;j<r->output.size();++j){std::ofstream bin(folder/("tensor-"+std::to_string(job->index)+"-"+std::to_string(j)+".f32"),std::ios::binary);bin.write(reinterpret_cast<const char*>(r->output[j].data()),r->output[j].size()*sizeof(float));require(bool(bin),"save tensor failed");}
   save_rgb(folder/("preview-"+std::to_string(f->id)+".png"),job->preview);
  },fail);
  int64_t start=now_us();
  std::thread producer([&]{try{
   std::unique_ptr<Camera> camera;cv::VideoCapture file;cv::Mat fixed;if(!o.image.empty()){auto bgr=cv::imread(o.image);require(!bgr.empty(),"image input missing");cv::cvtColor(bgr,fixed,cv::COLOR_BGR2RGB);}else if(!o.video.empty())require(file.open(o.video),"video input missing");else camera.reset(new Camera(o.width,o.height,o.memory));
   uint64_t id=0;
   while(running&&!stop_requested&&(!o.seconds||now_us()-start<int64_t(o.seconds)*1000000)){
    std::shared_ptr<Frame> f;if(camera)f=camera->take(o.project!="capture"&&o.capture_convert=="selected");else{f=std::make_shared<Frame>();if(!fixed.empty())f->rgb=fixed;else{cv::Mat bgr;if(!file.read(bgr))break;cv::cvtColor(bgr,f->rgb,cv::COLOR_BGR2RGB);}f->captured_us=now_us();f->id=++id;}
    if(!f)continue;
    ++captured;if(o.project=="capture"){if(f->id%std::max(1,o.save_every)==0)save_rgb(fs::path(o.out)/("frame-"+std::to_string(f->id)+".png"),f->rgb);}
    else{queue.put(f);if(!fixed.empty())std::this_thread::sleep_for(std::chrono::milliseconds(33));}
   }
  }catch(const std::exception&e){fail(e);}capture_done=true;});
  std::vector<std::thread> workers;
  if(o.project!="capture")for(int worker=0;worker<o.workers;++worker)workers.emplace_back([&,worker]{try{
   if(o.inference_cpus=="big"){cpu_set_t allowed,big;CPU_ZERO(&big);require(sched_getaffinity(0,sizeof(allowed),&allowed)==0,"read CPU affinity");int available=0;for(int cpu=4;cpu<8;++cpu)if(CPU_ISSET(cpu,&allowed)){CPU_SET(cpu,&big);++available;}require(available&&sched_setaffinity(0,sizeof(big),&big)==0,"RK3588 big CPU affinity unavailable");}
   auto& model=*models[worker];auto& rga=*resize[worker];bool stalled=false;
   while(running&&!stop_requested){auto f=queue.get();if(!f){if(capture_done)break;continue;}if(!stalled&&o.stall_ms&&now_us()-start>5000000){stalled=true;std::this_thread::sleep_for(std::chrono::milliseconds(o.stall_ms));}
    auto r=std::make_shared<Result>();r->id=f->id;r->frame=f;auto begin=now_us();materialize_rgb(*f,rga);if(o.depth_profile=="camera-rect")require(f->rgb.cols*9==f->rgb.rows*16,"camera-rect only accepts 16:9 input; use standard for other images");
    if(o.project=="real_esrgan"){int x=o.roi_x<0?(f->rgb.cols-256)/2:o.roi_x,y=o.roi_y<0?(f->rgb.rows-256)/2:o.roi_y;require(x>=0&&y>=0&&x+256<=f->rgb.cols&&y+256<=f->rgb.rows,"ROI outside input");r->prepared=f->rgb(cv::Rect(x,y,256,256)).clone();}
    else{r->lb=rga.letterbox(f->rgb,model.width,model.height);r->prepared=r->lb.rgb;}
    r->prep_ms=(now_us()-begin)/1000.;begin=now_us();r->output=model.run(r->prepared);r->infer_ms=(now_us()-begin)/1000.;++inferred;results.put(r);
   }
  }catch(const std::exception&e){fail(e);}});
  std::ofstream times(fs::path(o.out)/"frames.csv");times<<"id,captured_us,prep_ms,infer_ms,post_ms,display_ms,age_ms,outcome\n";
  std::ofstream tracks(fs::path(o.out)/"tracks.csv"),events(fs::path(o.out)/"events.csv");tracks<<"frame,time_us,class,id,x,y,w,h,score\n";events<<"frame,time_us,class,id,direction\n";
  std::map<int,std::unique_ptr<BYTETracker>> trackers;Counts counts;uint64_t shown=0,last_id=0,saved=0;int64_t last_track_us=0,last_seen=now_us();std::string hardware;int64_t hardware_time=0;
  Stats inference,age,preps,posts,display_cost;
  try{
   while(running&&!stop_requested){
    if(o.project=="capture"){if(capture_done)break;std::this_thread::sleep_for(std::chrono::milliseconds(50));continue;}
    auto r=results.get();if(!r){if(capture_done&&now_us()-last_seen>2000000)break;require(now_us()-last_seen<15000000,"no result for 15 seconds");continue;}
    last_seen=now_us();auto f=r->frame;if(f->id<=last_id){times<<f->id<<','<<f->captured_us<<','<<r->prep_ms<<','<<r->infer_ms<<",0,0,0,out_of_order\n";continue;}last_id=f->id;
    auto begin=now_us();cv::Mat canvas,save_depth,save_sr;bool save=o.save_every>0&&shown%o.save_every==0;
    if(o.project=="yolo11_bytetrack"){
     canvas=f->rgb.clone();auto det=decode_yolo(r->output,r->lb,canvas.cols,canvas.rows,.1);std::map<int,std::vector<Object>> grouped;
     for(auto&d:det)if(o.target_class<0||d.label==o.target_class)grouped[d.label].push_back({d.box,d.label,d.score});
     if(last_track_us&&f->captured_us-last_track_us>1000000){trackers.clear();counts.side.clear();counts.last.clear();}
     double dt=last_track_us?std::clamp(double(f->captured_us-last_track_us)/1000000.,.001,1.):1./30;last_track_us=f->captured_us;
     for(auto& kv:grouped)if(!trackers.count(kv.first))trackers[kv.first].reset(new BYTETracker(30,30));
     for(auto& kv:trackers){kv.second->set_timestep(dt*30);auto current=kv.second->update(grouped[kv.first]);for(auto& t:current){if(!t.is_activated)continue;auto box=cv::Rect2f(t.tlwh[0],t.tlwh[1],t.tlwh[2],t.tlwh[3]);cv::rectangle(canvas,box,cv::Scalar(0,255,0),2);cv::putText(canvas,"ID "+std::to_string(t.track_id),{int(box.x),std::max(30,int(box.y)-4)},cv::FONT_HERSHEY_SIMPLEX,.7,{0,255,0},2);int event=counts.update(t.track_id,(box.y+box.height)/canvas.rows,f->captured_us,o.line);if(event)events<<f->id<<','<<f->captured_us<<','<<kv.first<<','<<t.track_id<<','<<event<<'\n';tracks<<f->id<<','<<f->captured_us<<','<<kv.first<<','<<t.track_id<<','<<box.x<<','<<box.y<<','<<box.width<<','<<box.height<<','<<t.score<<'\n';}}
     counts.expire(f->captured_us);cv::line(canvas,{0,int(canvas.rows*o.line)},{canvas.cols-1,int(canvas.rows*o.line)},{255,200,0},2);cv::putText(canvas,"IN "+std::to_string(counts.forward)+" OUT "+std::to_string(counts.backward),{20,150},cv::FONT_HERSHEY_SIMPLEX,1,{255,255,255},2);
    }else if(o.project=="depth_anything_v2"){
     int w=models[0]->width,h=models[0]->height;require(r->output.size()==1&&r->output[0].size()==size_t(w*h),"depth output shape");cv::Mat depth(h,w,CV_32F,r->output[0].data());require(cv::checkRange(depth),"depth NaN/Inf");cv::Mat restored;cv::resize(depth(cv::Rect(r->lb.x,r->lb.y,r->lb.w,r->lb.h)),restored,f->rgb.size(),0,0,cv::INTER_LINEAR);cv::Mat norm,color;cv::normalize(restored,norm,0,255,cv::NORM_MINMAX,CV_8U);cv::applyColorMap(norm,color,cv::COLORMAP_INFERNO);cv::cvtColor(color,color,cv::COLOR_BGR2RGB);cv::Mat left;cv::resize(f->rgb,left,{960,540});cv::resize(color,color,{960,540});cv::hconcat(left,color,canvas);
     if(save)save_depth=restored;
    }else{
     require(r->output.size()==1&&r->output[0].size()==3*1024*1024,"SR output shape");cv::Mat sr=rgb_from_planar(r->output[0],1024,1024,o.render=="vector");cv::Mat baseline;cv::resize(r->prepared,baseline,{1024,1024},0,0,cv::INTER_CUBIC);cv::hconcat(baseline,sr,canvas);if(save)save_sr=sr;
    }
    double post=(now_us()-begin)/1000.;double seconds=(now_us()-start)/1000000.;if(now_us()-hardware_time>1000000){hardware=metrics();hardware_time=now_us();}
    std::ostringstream text;text<<o.project<<" processed="<<std::fixed<<std::setprecision(1)<<(shown+1)/seconds<<"fps infer="<<r->infer_ms<<"ms age="<<(now_us()-f->captured_us)/1000.<<"ms display="<<(display?(shown+1)/seconds:0)<<"fps";
    cv::putText(canvas,text.str(),{20,38},cv::FONT_HERSHEY_SIMPLEX,.75,{0,0,0},4);cv::putText(canvas,text.str(),{20,38},cv::FONT_HERSHEY_SIMPLEX,.75,{255,255,255},1);cv::putText(canvas,hardware,{20,75},cv::FONT_HERSHEY_SIMPLEX,.6,{0,0,0},4);cv::putText(canvas,hardware,{20,75},cv::FONT_HERSHEY_SIMPLEX,.6,{255,255,255},1);
    if(o.display_error_after&&now_us()-start>int64_t(o.display_error_after)*1000000)throw std::runtime_error("injected display failure; not physical unplug");
    begin=now_us();int64_t finished=display?display->show(canvas):now_us();double dms=(now_us()-begin)/1000.,ams=(finished-f->captured_us)/1000.;
    if(save){auto job=std::make_shared<SaveJob>();job->result=r;job->relative_depth=save_depth;job->sr=save_sr;job->preview=canvas;job->index=saved++;saver.submit(job);}
    ++shown;inference.add(r->infer_ms);age.add(ams);preps.add(r->prep_ms);posts.add(post);display_cost.add(dms);times<<f->id<<','<<f->captured_us<<','<<r->prep_ms<<','<<r->infer_ms<<','<<post<<','<<dms<<','<<ams<<",processed\n";
    std::ofstream st(fs::path(o.out)/"status.json.tmp");st<<"{\"captured\":"<<captured<<",\"inferred\":"<<inferred<<",\"processed\":"<<shown<<",\"elapsed_s\":"<<seconds<<",\"fps\":"<<shown/seconds<<",\"age_ms\":"<<ams<<",\"in\":"<<counts.forward<<",\"out\":"<<counts.backward<<"}\n";st.close();fs::rename(fs::path(o.out)/"status.json.tmp",fs::path(o.out)/"status.json");
   }
  }catch(const std::exception&e){fail(e);}
  running=false;queue.close();results.close();producer.join();for(auto& t:workers)t.join();saver.finish();display.reset();double elapsed=(now_us()-start)/1000000.;
  std::ofstream summary(fs::path(o.out)/"summary.json");summary<<"{\"project\":\""<<o.project<<"\",\"seconds\":"<<elapsed<<",\"captured\":"<<captured<<",\"inferred\":"<<inferred<<",\"processed\":"<<shown<<",\"fps\":"<<shown/elapsed<<",\"raw_replaced\":"<<queue.replaced<<",\"results_replaced\":"<<results.replaced<<",\"out_of_order_results\":"<<results.out_of_order<<",\"in\":"<<counts.forward<<",\"out\":"<<counts.backward<<",\"error\":"<<(!error.empty()?"true":"false")<<",\"latency_scope\":\"appsink retrieval to flip or processing completion; not optical\"";
  summary<<",\"save_jobs_submitted\":"<<saver.submitted.load()<<",\"save_jobs_written\":"<<saver.written.load()<<",\"save_jobs_replaced\":"<<saver.replaced.load()<<",\"save_jobs_failed\":"<<saver.errors.load()<<",\"save_jobs_discarded\":"<<saver.discarded.load()<<",\"save_wall_ms_total\":"<<saver.total_us.load()/1000.<<",\"save_wall_ms_max\":"<<saver.max_us.load()/1000.;
  auto stats=[&](const char* name,const Stats& v){summary<<",\""<<name<<"\":{\"p50\":"<<v.quantile(.5)<<",\"p95\":"<<v.quantile(.95)<<",\"p99\":"<<v.quantile(.99)<<",\"max\":"<<v.max()<<'}';};stats("inference_ms",inference);stats("age_ms",age);stats("preprocess_ms",preps);stats("postprocess_ms",posts);stats("display_ms",display_cost);summary<<"}\n";summary.close();
  if(!error.empty()){std::ofstream(fs::path(o.out)/"error.txt")<<error;std::cerr<<error<<'\n';return 1;}return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
