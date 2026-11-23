#include "dma_chain_model.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using Model = cxxrtl_design::p_dma__burst__chain;
struct Transfer {uint32_t from,to,length;};
constexpr uint32_t burst_bytes = CHAIN_BURST_BYTES;

void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}

struct Harness {
    Model dut;
    uint32_t source=0,destination=0,length=0,status=0;
    unsigned pending=0;
    std::vector<Transfer> launches;

    Harness() {
        dut.p_clk.set<uint32_t>(0);dut.p_reset__n.set<uint32_t>(0);
        tick(0);
        dut.p_reset__n.set<uint32_t>(1);
        tick(0);
    }
    uint32_t tick(uint32_t address,uint32_t value=0,bool write=false) {
        dut.p_clk.set<uint32_t>(0);dut.p_addr.set<uint32_t>(address);
        dut.p_cs.set<uint32_t>(1);dut.p_wr__n.set<uint32_t>(!write);
        dut.p_wdata.set<uint32_t>(value);dut.step();
        const auto reg=dut.p_child__addr.get<uint32_t>();
        const bool writing=dut.p_child__cs.get<uint32_t>() && !dut.p_child__wr__n.get<uint32_t>();
        const auto payload=dut.p_child__wdata.get<uint32_t>();
        // Read-data arrives one cycle after the child CSR address is sampled.
        const uint32_t response=reg==0 ? status : reg==3 ? length : 0;
        if(pending && --pending==0)status=1;
        if(writing) {
            if(reg==0)status=0;
            if(reg==1)source=payload;
            if(reg==2)destination=payload;
            if(reg==3)length=payload;
            if(reg==6 && (payload&8)) {
                require(payload==0x888u,"Unexpected child control word");
                require(!pending && status==0,"Overlapping command or stale done");
                require(length && length<=burst_bytes && !(length%16),"Invalid child burst length");
                launches.push_back({source,destination,length});pending=5+unsigned(launches.size()%9);status=2;
            }
        }
        dut.p_clk.set<uint32_t>(1);dut.step();
        dut.p_child__rdata.set<uint32_t>(response);dut.step();
        return dut.p_rdata.get<uint32_t>();
    }
    void write(unsigned reg,uint32_t value) {tick(reg,value,true);}
    void start(uint32_t bytes,uint32_t from=0x30021000,uint32_t to=0x1000) {
        write(0,0);write(1,from);write(2,to);write(3,bytes);write(6,0x2888);
    }
    uint32_t finish() {
        for(unsigned i=0;i<12000;++i) {
            const auto state=tick(0);
            if(state&5)return state;
        }
        throw std::runtime_error("Timed out polling simulated DMA");
    }
};

int main() {
    try {
        Harness h;
        require(h.tick(4)==0x444d4331u,"Capability read failed");
        const std::vector<uint32_t> lengths{16,burst_bytes,burst_bytes+16,2*burst_bytes,32768,65536};
        for(uint32_t bytes : lengths) {
            h.launches.clear();h.start(bytes);
            require(h.finish()==1,"Chain failed");
            uint32_t offset=0;
            for(const auto& x : h.launches) {
                const auto n=std::min(burst_bytes,bytes-offset);
                require(x.from==0x30021000u+offset && x.to==0x1000u+offset && x.length==n,"Incorrect burst coverage");
                offset+=n;
            }
            require(offset==bytes && h.tick(3)==0,"Incomplete chain");
            h.write(6,0);
        }
        for(uint32_t length : {0u,17u,65552u}) {
            h.launches.clear();h.start(length);
            require(h.finish()==4 && h.launches.empty(),"Invalid length reached DMA core");
            h.write(6,0);
        }
        h.start(16,0x30021001u);
        require(h.finish()==4 && h.launches.empty(),"Unaligned source reached DMA core");
        h.write(6,0);
        h.start(2*burst_bytes);
        while(h.launches.empty())h.tick(0);
        h.write(6,0);
        for(unsigned i=0;i<100;++i)h.tick(0);
        require(h.launches.size()==1 && !h.pending,"Abort launched another burst");
        h.launches.clear();h.start(16);require(h.finish()==1,"Cannot restart after abort");h.write(6,0);
        h.launches.clear();h.write(0,0);h.write(1,0x30021000u);h.write(2,0x1000);h.write(3,16);h.write(6,0x888);
        require(h.finish()==1 && h.launches.size()==1,"Legacy command failed");
        std::cout<<"PASS DMA chain, burst="<<burst_bytes<<", aligned tails, full window, rejects, abort, legacy\n";
    } catch(const std::exception& e) {
        std::cerr<<e.what()<<'\n';return 1;
    }
}
