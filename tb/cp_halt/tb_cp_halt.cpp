#include "Vtb_cp_halt_top.h"
#include "verilated.h"
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
struct Bench {
 Vtb_cp_halt_top dut;
 std::deque<uint32_t> pending;
 unsigned cycles=0,issues=0,reads=0,responses=0,writes=0;
 void check(bool ok,const char* msg) {
  if(!ok)throw std::runtime_error("cycle"+std::to_string(cycles)+": "+msg);
 }
 void tick(bool ready,bool response) {
  dut.clk=0;dut.dram_ready=ready;dut.dram_rsp=response;
  check(!response||!pending.empty(),"test returned unissued read");
  for(unsigned i=0;i<16;++i)dut.response_data[i]=0;
  if(response) {
   // Real DMA then HALT. Any later fetch is a poison NOP which must be discarded.
   const uint32_t addr=pending.front();
   dut.response_data[0]=0xFFFF0000u|(addr==0?0x10:addr==64?0x01:0x00);
  }
  dut.eval();
  bool request=dut.req_valid&&ready;
  if(dut.instr_issued)++issues;
  if(response){check(dut.cp_rsp,"fetch response not routed to CP");pending.pop_front();++responses;}
  if(request) {
   if(dut.req_we){check(dut.req_addr==4096,"wrong DMA write selected");++writes;}
   else {
    check(dut.cp_valid&&dut.cp_ready,"phantom CP fetch accepted");
    pending.push_back(dut.req_addr);++reads;
   }
  }
  dut.clk=1;dut.eval();++cycles;
 }
 Bench() {
  dut.rst_n=0;dut.start=0;dut.engine_busy=0;dut.dma_req_valid=0;
  tick(false,false);dut.rst_n=1;dut.start=1;tick(false,false);dut.start=0;
 }
};
static void run(bool acceptWithHalt,bool extraTail,unsigned stallCycles) {
 Bench b;
 b.tick(true,false);       // accept DMA instruction fetch at0
 b.tick(true,true);        // return DMA, accept HALT fetch at64
 b.dut.engine_busy=1;      // DMA execution may remain active after HALT
 if(extraTail)b.tick(true,false); // already accepted post-HALT prefetch128
 b.tick(false,false);      // next CP fetch is presented and held in arb
 b.check(b.dut.fetch_held,"failed to establish held fetch");
 b.tick(acceptWithHalt,true); // HALT response arrives while next fetch is pending
 b.dut.dma_req_valid=1;    // competing engine requests must not create phantom fetches
 if(!acceptWithHalt) {
  b.check(b.dut.cp_valid&&b.dut.fetch_held,"HALT withdrew an unaccepted fetch");
  if(extraTail)b.tick(false,true); // discard older tail while held fetch still pending
  for(unsigned i=0;i<stallCycles;++i) {
   b.tick(false,false);
   b.check(b.dut.cp_valid&&!b.dut.done,"held fetch lost or completion premature");
  }
  b.tick(true,false);     // accept held fetch after HALT has been observed
 }
 // Outstanding fetches, including the accepted-after-HALT request, must drain.
 bool wrote=false;
 for(unsigned i=0;i<20 && (!b.pending.empty()||!wrote);++i) {
  auto oldWrites=b.writes;
  b.tick(true,!b.pending.empty());
  if(b.writes!=oldWrites){wrote=true;b.dut.dma_req_valid=0;}
 }
 b.dut.engine_busy=0;
 for(unsigned i=0;i<10&&!b.dut.done;++i)b.tick(true,false);
 b.check(b.dut.done,"CP failed to complete after held post-HALT fetch drained");
 b.check(b.dut.outstanding==0&&b.dut.discard==0&&!b.dut.fetch_held,"post-HALT accounting did not drain");
 b.check(b.issues==2,"post-HALT fetched instruction was executed");
 b.check(b.reads==b.responses,"fetch request/response count mismatch");
}
int main() {
 try {
  for(bool same:{false,true})for(bool tail:{false,true})for(unsigned delay:{0u,1u,200u})run(same,tail,delay);
  std::cout<<"cp_halt PASS (12 held-fetch/HALT/response-interleave scenarios)\n";
 }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
