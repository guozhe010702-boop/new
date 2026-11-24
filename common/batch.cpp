#include "core.hpp"
#include <filesystem>
#include <iostream>
namespace fs=std::filesystem;
using namespace aa;
int main(int argc,char**argv){try{
 require(argc==4||(argc==5&&std::string(argv[4])=="sr-cpu-tail"),"Usage: model_batch MODEL INPUT_DIR OUTPUT_DIR [sr-cpu-tail]");require(!fs::exists(argv[3]),"output must be new");fs::create_directories(argv[3]);Model model(argv[1],7,argc==5);std::cout<<model.sdk<<'\n';std::ofstream csv(fs::path(argv[3])/"inference.csv");csv<<"file,infer_ms\n";
 std::vector<fs::path> paths;for(auto& p:fs::directory_iterator(argv[2]))if(p.path().extension()==".png"||p.path().extension()==".jpg")paths.push_back(p.path());std::sort(paths.begin(),paths.end());
 for(auto& p:paths){auto bgr=cv::imread(p.string());require(!bgr.empty(),"invalid input image");cv::Mat rgb;cv::cvtColor(bgr,rgb,cv::COLOR_BGR2RGB);auto start=now_us();auto outputs=model.run(rgb);csv<<p.filename().string()<<','<<(now_us()-start)/1000.<<'\n';for(size_t j=0;j<outputs.size();++j){auto name=fs::path(argv[3])/(p.stem().string()+"-"+std::to_string(j)+".f32");std::ofstream f(name,std::ios::binary);f.write(reinterpret_cast<const char*>(outputs[j].data()),outputs[j].size()*4);}csv.flush();}
 require(!paths.empty(),"no input images");return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
