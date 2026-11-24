#include "core.hpp"
#include "BYTETracker.h"
#include <iostream>
using namespace aa;
int main(){
 Counts c;require(c.update(1,.4,100)==0,"first observation must not count");require(c.update(1,.499,200)==0,"hysteresis");require(c.update(1,.6,300)==1,"forward crossing");require(c.update(1,.59,400)==0,"same side duplicate");require(c.update(1,.4,500)==-1,"reverse crossing");require(c.forward==1&&c.backward==1,"counts");c.expire(20000000);require(c.side.empty(),"retired track state leak");
 Latest<Frame> q;auto a=std::make_shared<Frame>();a->id=1;q.put(a);auto b=std::make_shared<Frame>();b->id=2;q.put(b);q.put(a);require(q.replaced==1&&q.out_of_order==1&&q.get()->id==2,"latest frame behavior and out-of-order result rejection");q.close();
 BYTETracker tracker(30,30);std::vector<STrack> tracks;for(int i=0;i<20;++i){tracker.set_timestep(.5+i*.05);tracks=tracker.update({{{float(100+i),100,50,80},0,.95f}});}require(tracks.size()==1&&tracks[0].is_activated,"ByteTrack stable object");int id=tracks[0].track_id;tracks=tracker.update({});tracks=tracker.update({{{121,100,50,80},0,.95f}});require(tracks.size()==1&&tracks[0].track_id==id,"ByteTrack short occlusion ID");
 Stats hist;hist.add(10.01);hist.add(20);require(hist.quantile(.5)==10.1&&hist.quantile(.95)==20,"histogram upper-edge quantiles");hist.add(3000);require(hist.max()==3000,"overflow maximum");
 BYTETracker moving(30,30);Counts crossing;for(int i=0;i<120;++i){float y=i<60?50+i*3:230-(i-60)*3;auto ts=moving.update({{{100,y,50,60},0,.95f}});for(auto&t:ts)if(t.is_activated)crossing.update(t.track_id,(t.tlwh[1]+t.tlwh[3])/400.,i*33333,.5);}require(crossing.forward==1&&crossing.backward==1,"ByteTrack synthetic bidirectional crossing count");
 std::cout<<"core contracts passed\n";
}
