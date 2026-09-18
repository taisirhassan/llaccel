// Independent ordered-response scoreboard; no reference arbiter RTL is reused.
#include "Vdram_arb.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <deque>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>

struct Request { unsigned owner; uint32_t serial; bool write; };
struct Pending { Request request; uint64_t due; };
class Bench {
 public:
  Vdram_arb dut;
  std::mt19937 rng;
  std::array<std::optional<Request>,3> active;
  std::deque<Pending> pending;
  std::optional<Request> held;
  std::array<uint64_t,3> accepted{}, returned{}, writes{};
  uint64_t cycle=0, blocked=0, fullCycles=0, simultaneous=0;
  unsigned serial=1, last=2;
  explicit Bench(unsigned seed):rng(seed) { reset(); }
  void check(bool condition, const std::string& message) {
    if(!condition) throw std::runtime_error("cycle "+std::to_string(cycle)+": "+message);
  }
  static uint32_t address(Request r) { return r.serial*64; }
  static uint32_t data(Request r, unsigned word) { return 0x913acdefu ^ (r.serial*65537u) ^ (word*1234567u); }
  static uint64_t mask(Request r) { return 0xa55a5aa512349876ull ^ (uint64_t(r.serial)*0x101010101ull); }
  void reset() {
    dut.clk=0;dut.rst_n=0;dut.dma_req_valid=0;dut.attn_req_valid=0;dut.cp_req_valid=0;
    dut.dram_rsp_valid=0;dut.dram_req_ready=0;dut.eval();dut.clk=1;dut.eval();
    dut.clk=0;dut.rst_n=1;dut.eval();
    active={};pending.clear();held.reset();last=2;
    check(!dut.cp_rsp_valid && !dut.dma_rsp_valid && !dut.attn_rsp_valid,"reset leaked response");
  }
  void offer(unsigned owner, bool write) {
    check(!active[owner],"test overwrote unaccepted request");
    active[owner]=Request{owner,serial++,write && owner!=0};
  }
  void tick(bool ready, bool responses, unsigned latency=3) {
    dut.clk=0;
    dut.cp_req_valid=bool(active[0]);dut.dma_req_valid=bool(active[1]);dut.attn_req_valid=bool(active[2]);
    for(unsigned owner=0;owner<3;owner++) {
      // Inactive buses deliberately change, proving arbitration locks the owner.
      Request r=active[owner].value_or(Request{owner,unsigned(cycle+10000+owner),false});
      if(owner==0) dut.cp_req_addr=address(r);
      if(owner==1) { dut.dma_req_addr=address(r);dut.dma_req_we=r.write;dut.dma_req_wstrb=mask(r);
        for(unsigned i=0;i<16;i++)dut.dma_req_wdata[i]=data(r,i); }
      if(owner==2) { dut.attn_req_addr=address(r);dut.attn_req_we=r.write;dut.attn_req_wstrb=mask(r);
        for(unsigned i=0;i<16;i++)dut.attn_req_wdata[i]=data(r,i); }
    }
    bool response=responses && !pending.empty() && pending.front().due<=cycle;
    dut.dram_req_ready=ready;dut.dram_rsp_valid=response;
    for(unsigned i=0;i<16;i++)dut.dram_rsp_rdata[i]=response ? data(pending.front().request,i) : 0;
    dut.eval();
    std::array<bool,3> rsp{bool(dut.cp_rsp_valid),bool(dut.dma_rsp_valid),bool(dut.attn_rsp_valid)};
    unsigned rspCount=rsp[0]+rsp[1]+rsp[2];
    check(rspCount==unsigned(response),"lost, duplicate, or unsolicited response");
    if(response) {
      Request r=pending.front().request;
      check(rsp[r.owner],"ordered response routed to wrong owner");
      for(unsigned i=0;i<16;i++)check(dut.rsp_rdata[i]==data(r,i),"response data corruption");
      returned[r.owner]++;
    }
    std::array<bool,3> handshakes{bool(dut.cp_req_ready)&&bool(active[0]),
      bool(dut.dma_req_ready)&&bool(active[1]),bool(dut.attn_req_ready)&&bool(active[2])};
    unsigned count=handshakes[0]+handshakes[1]+handshakes[2];
    check(count==unsigned(dut.dram_req_valid && ready),"request handshake duplicated or lost");
    if(pending.size()==TEST_TAG_DEPTH) {
      fullCycles++;check(!dut.dram_req_valid,"tag queue overcommitted");
    }
    std::optional<Request> issued;
    if(dut.dram_req_valid) {
      std::optional<Request> expected=held;
      if(!expected) for(unsigned step=1;step<=3;step++) {
        unsigned owner=(last+step)%3;
        if(active[owner]) { expected=active[owner];break; }
      }
      check(bool(expected),"valid output with no requester");
      Request r=*expected;
      check(dut.dram_req_addr==address(r),"round-robin/held-owner address mismatch");
      check(bool(dut.dram_req_we)==r.write,"read/write mismatch");
      if(r.write) {
        check(dut.dram_req_wstrb==mask(r),"write mask changed under backpressure");
        for(unsigned i=0;i<16;i++)check(dut.dram_req_wdata[i]==data(r,i),"write data changed under backpressure");
      }
      if(ready) {check(handshakes[r.owner],"ready routed to wrong requester");issued=r;held.reset();last=r.owner;}
      else {held=r;blocked++;}
    } else check(!held,"valid deasserted while a presented request was held");
    // Responses can only belong to old accepted reads, never this cycle's request.
    if(response) pending.pop_front();
    if(issued) {
      auto r=*issued;accepted[r.owner]++;
      if(r.write)writes[r.owner]++;
      else pending.push_back({r,cycle+latency});
      active[r.owner].reset();
    }
    if(response && issued && !issued->write)simultaneous++;
    dut.clk=1;dut.eval();cycle++;
  }
  void drain() {
    for(unsigned tries=0; tries<10000 && (active[0]||active[1]||active[2]||!pending.empty()); tries++)tick(true,true);
    check(!active[0]&&!active[1]&&!active[2]&&pending.empty(),"drain deadlocked");
  }
};
int main(int argc,char**argv) {
  try {
    unsigned seed=argc>1 ? std::stoul(argv[1]):1;
    Bench b(seed);
    // First CP is held; newly arriving DMA/attention cannot preempt it.
    b.offer(0,false);b.tick(false,false);
    b.offer(1,true);b.offer(2,true);
    for(unsigned i=0;i<200;i++)b.tick(false,false);
    b.drain();
    // Persistent contenders: accepted request sequence must be exactly round robin.
    for(unsigned i=0;i<1200;i++) {
      for(unsigned owner=0;owner<3;owner++)if(!b.active[owner])b.offer(owner,i%3==0);
      b.tick(true,true,1);
    }
    b.drain();
    for(unsigned owner=0;owner<3;owner++)b.check(b.accepted[owner]>100,"persistent requester starved");
    // Fill every owner-tag slot while responses are stalled, then overlap pop/push.
    for(unsigned i=0;i<TEST_TAG_DEPTH+20;i++) {
      for(unsigned owner=0;owner<3;owner++)if(!b.active[owner])b.offer(owner,false);
      b.tick(true,false,1);
    }
    b.check(b.pending.size()==TEST_TAG_DEPTH,"failed to exercise full tags");
    b.drain();
    for(unsigned i=0;i<50000;i++) {
      for(unsigned owner=0;owner<3;owner++)if(!b.active[owner] && b.rng()%100<75)b.offer(owner,b.rng()%2);
      bool ready=i%1000>=150 && b.rng()%100<65;
      bool responses=i%777>=200 && b.rng()%100<70;
      b.tick(ready,responses,1+b.rng()%25);
    }
    b.drain();
    for(unsigned owner=0;owner<3;owner++) {
      b.check(b.returned[owner]+b.writes[owner]==b.accepted[owner],"request/response accounting mismatch");
      b.check(b.returned[owner]>100,"insufficient owner response coverage");
    }
    b.check(b.writes[1]>100 && b.writes[2]>100,"insufficient mixed write coverage");
    b.check(b.blocked>200 && b.fullCycles>10,"insufficient backpressure/saturation coverage");
    if(TEST_TAG_DEPTH>1)b.check(b.simultaneous>10,"no simultaneous tag push/pop coverage");
    // Reset drops outstanding ownership tags and held requests; new transactions recover.
    b.offer(2,false);b.tick(true,false);b.offer(1,true);b.tick(false,false);b.reset();
    b.offer(0,false);b.offer(1,true);b.offer(2,false);b.drain();
    std::cout<<"dram_arb PASS depth="<<TEST_TAG_DEPTH<<" seed="<<seed<<" cycles="<<b.cycle
      <<" accepted="<<b.accepted[0]+b.accepted[1]+b.accepted[2]<<" stalls="<<b.blocked
      <<" full="<<b.fullCycles<<" simultaneous="<<b.simultaneous<<"\n";
  } catch(const std::exception&e) {std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
