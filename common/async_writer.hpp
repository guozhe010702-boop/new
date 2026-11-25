#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
namespace aa {
// One in-flight write plus one latest candidate. The producer never waits for IO.
template<class T> class AsyncLatestWriter {
 using Ptr=std::shared_ptr<T>;
 std::mutex mutex;std::condition_variable changed;Ptr pending;bool closing=false;
 std::function<void(const Ptr&)> write;std::function<void(const std::exception&)> failed;
 std::thread worker;
 void run(){
  for(;;){Ptr item;{std::unique_lock<std::mutex> lock(mutex);changed.wait(lock,[&]{return closing||pending;});if(!pending)return;item=std::move(pending);}
   auto begin=std::chrono::steady_clock::now();bool good=true;
   try{write(item);++written;}catch(const std::exception& error){++errors;good=false;failed(error);}catch(...){++errors;good=false;std::runtime_error error("unknown asynchronous save failure");failed(error);}
   auto elapsed=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-begin).count();total_us+=elapsed;auto maximum=max_us.load();while(uint64_t(elapsed)>maximum&&!max_us.compare_exchange_weak(maximum,elapsed)){}
   if(!good){std::lock_guard<std::mutex> lock(mutex);closing=true;if(pending){++discarded;pending.reset();}return;}
  }
 }
public:
 std::atomic<uint64_t> submitted{0},replaced{0},written{0},errors{0},discarded{0},total_us{0},max_us{0};
 AsyncLatestWriter(std::function<void(const Ptr&)> action,std::function<void(const std::exception&)> on_error):write(std::move(action)),failed(std::move(on_error)){worker=std::thread([this]{run();});}
 AsyncLatestWriter(const AsyncLatestWriter&)=delete;AsyncLatestWriter& operator=(const AsyncLatestWriter&)=delete;
 ~AsyncLatestWriter(){finish();}
 bool submit(Ptr item){std::lock_guard<std::mutex> lock(mutex);++submitted;if(closing){++discarded;return false;}if(pending)++replaced;pending=std::move(item);changed.notify_one();return true;}
 void finish(){{std::lock_guard<std::mutex> lock(mutex);closing=true;}changed.notify_one();if(worker.joinable())worker.join();}
};
}
