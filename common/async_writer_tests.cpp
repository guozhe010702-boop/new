#include "async_writer.hpp"
#include <future>
#include <iostream>
#include <vector>
static void check(bool condition,const char* error){if(!condition)throw std::runtime_error(error);}
int main(){try{
 std::promise<void> began,release;auto started=began.get_future();auto gate=release.get_future().share();std::vector<int> order;std::string error;
 aa::AsyncLatestWriter<int> writer([&](const std::shared_ptr<int>& item){if(*item==1){began.set_value();gate.wait();}order.push_back(*item);},[&](const std::exception& e){error=e.what();});
 writer.submit(std::make_shared<int>(1));started.wait();writer.submit(std::make_shared<int>(2));writer.submit(std::make_shared<int>(3));writer.submit(std::make_shared<int>(4));release.set_value();writer.finish();check(order==std::vector<int>({1,4}),"latest pending save not retained");check(writer.submitted==4&&writer.written==2&&writer.replaced==2&&writer.errors==0&&error.empty(),"save accounting");check(!writer.submit(std::make_shared<int>(5))&&writer.discarded==1,"closed writer accepted work");
 std::string failure;aa::AsyncLatestWriter<int> broken([](const std::shared_ptr<int>&){throw std::runtime_error("injected IO failure");},[&](const std::exception& e){failure=e.what();});broken.submit(std::make_shared<int>(1));broken.finish();check(broken.errors==1&&broken.written==0&&failure=="injected IO failure","IO error was not propagated");std::cout<<"bounded async writer, drain, replacement and error propagation passed\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
