#include "Vtb_xbar_top.h"
#include "Vtb_xbar_top___024root.h"
#include "tb_common.h"
#include "verilated.h"
using namespace tb;
int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Rng rng(seedFromArgs(argc, argv));
  Vtb_xbar_top top;
  SramImage ram; TB_COLLECT_BANKS(top, tb_xbar_top, ram);
  auto edge = [&] { top.clk=1; top.eval(); top.clk=0; top.eval(); };
  top.clk=0; top.rst_n=0; top.valid=0; top.eval(); edge();
  top.rst_n=1; edge();
  std::vector<uint8_t> ref(SramImage::kBytes); rng.fill(ref.data(), ref.size()); ram.fill(ref);
  constexpr unsigned writes = (1<<2)|(1<<4)|(1<<7)|(1<<8);
  unsigned held=0, grants[10]={};
  Report rep;
  for (int cycle=0; cycle<10000; ++cycle) {
    for (int p=0;p<10;++p) if (!(held>>p&1)) {
      top.valid=(top.valid&~(1<<p))|(rng.coin(65)<<p);
      unsigned sz=p==0?3:rng.range(0,2), n=sz==3?256:16<<sz;
      top.size[p]=sz;
      top.addr[p]=256*rng.range(0,15)+16*rng.range(0,(256-n)/16);
      rng.fill(reinterpret_cast<uint8_t*>(top.wdata[p].data()),64);
      top.wstrb[p]=rng.u64();
    }
    top.eval(); unsigned taken=0, expected=0, rd=0, wr=0;
    std::vector<uint8_t> reads[10];
    for (int p=0;p<10;++p) {
      unsigned n=top.size[p]==3?256:16<<top.size[p];
      unsigned mask=((1u<<(n/16))-1)<<((top.addr[p]>>4)&15);
      if ((top.valid>>p&1)&&!(taken&mask)) {
        taken|=mask; expected|=1<<p; ++grants[p];
        if (writes>>p&1) {
          auto* data=reinterpret_cast<uint8_t*>(top.wdata[p].data());
          for (unsigned j=0;j<n;++j) if (top.wstrb[p]>>j&1) {ref[top.addr[p]+j]=data[j];++wr;}
        } else { rd+=n; reads[p].assign(ref.begin()+top.addr[p],ref.begin()+top.addr[p]+n); }
      }
    }
    bool ok=top.grant==expected && top.stall==(top.valid&~expected) && top.rd_bytes==rd && top.wr_bytes==wr;
    held=top.valid&~expected; edge();
    ok &= top.rvalid==(expected&~writes);
    for(int p=0;p<10;++p) if(!reads[p].empty())
      ok &= std::memcmp(top.rdata[p].data(),reads[p].data(),reads[p].size())==0;
    if (!ok) {rep.tally(false,"arbitration/data/counter mismatch"); break;}
    rep.tally(true,"cycle");
  }
  auto got=ram.snapshot(); rep.tally(got==ref,"whole SRAM byte-enable comparison");
  for(int p=0;p<10;++p) rep.tally(grants[p]>0,"every port must receive grants");
  top.final(); return rep.finish("tb_xbar");
}
