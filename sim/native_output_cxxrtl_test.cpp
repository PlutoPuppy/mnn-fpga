#include "native_model.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using Model=cxxrtl_design::p_output__loader__native;
int partial(int band,int block,int lane,int column) {
    return column==0 ? (band%2==0 ? 50000 : -50000) :
        (band*119+block*37+lane*53+column*71)%2001-1000;
}
template<size_t Width> unsigned bits(const cxxrtl::value<Width>& value,unsigned offset,unsigned count) {
    uint64_t word=value.data[offset/32];
    if(offset%32+count>32)word|=uint64_t(value.data[offset/32+1])<<32;
    return unsigned((word>>(offset%32))&((uint64_t(1)<<count)-1));
}
int main() {
    try {
        Model dut;
        std::array<int,128> coefficients{},biases{};
        int bands=1,block=0,columns=16,pitch=16,valid_rows=16,band=0;
        size_t checks=0,cases=0;
        bool monitor=false;
        bool lookup=false;
        std::vector<bool> seen;
        auto tick=[&]() {
            dut.p_clock.set<uint32_t>(0);dut.step();
            if(monitor)for(int lane=0;lane<16;++lane)if(bits(dut.p_obuf__wren,lane,1)) {
                const int col=int(bits(dut.p_obuf__waddr,lane*12,12))-block*pitch;
                if(band!=bands-1 || lane>=valid_rows || col<0 || col>=columns)
                    throw std::runtime_error("Early write or invalid output address");
                const size_t index=size_t(lane)*columns+col;
                if(seen[index])throw std::runtime_error("Duplicate result");
                seen[index]=true;
                int64_t sum=0;
                for(int k=0;k<bands;++k)sum+=partial(k,block,lane,col);
                const int64_t value=sum*coefficients[col]+int64_t(biases[col])*256;
                const int64_t rounded=value>=0 ? (value+8388608)/16777216 : -((-value+8388608)/16777216);
                int expected=int(std::clamp(rounded,int64_t(-127),int64_t(127)));
                if(lookup) expected=int(int8_t(uint8_t(expected)^0x5a));
                const int actual=int(int8_t(bits(dut.p_obuf__wdata,lane*8,8)));
                if(actual!=expected) {
                    std::cerr<<"bands="<<bands<<" block="<<block<<" lane="<<lane<<" column="<<col
                             <<" actual="<<actual<<" expected="<<expected<<'\n';
                    throw std::runtime_error("Native output mismatch");
                }
                ++checks;
            }
            dut.p_clock.set<uint32_t>(1);dut.step();
            dut.p_clock.set<uint32_t>(0);dut.step();
        };
        dut.p_rst__n.set<uint32_t>(0);for(int i=0;i<4;++i)tick();
        dut.p_rst__n.set<uint32_t>(1);
        for(int col=0;col<128;++col) {
            coefficients[col]=col==0 ? 4194304 : 10000+col*37;
            biases[col]=(col%9-4)*32768;
        }
        for(int col=0;col<128;col+=2) {
            dut.p_cfg__write.set<uint32_t>(1);dut.p_cfg__index.set<uint32_t>(col);
            dut.p_cfg__data.data[0]=coefficients[col];dut.p_cfg__data.data[1]=uint32_t(biases[col]);
            dut.p_cfg__data.data[2]=coefficients[col+1];dut.p_cfg__data.data[3]=uint32_t(biases[col+1]);tick();
        }
        dut.p_cfg__write.set<uint32_t>(0);
        for(int word=0;word<16;++word) {
            dut.p_lut__write.set<uint32_t>(1);dut.p_lut__index.set<uint32_t>(word);
            for(int w=0;w<4;++w) {
                uint32_t packed=0;
                for(int b=0;b<4;++b)packed|=uint32_t((word*16+w*4+b)^0x5a)<<(b*8);
                dut.p_lut__data.data[w]=packed;
            }
            tick();
        }
        dut.p_lut__write.set<uint32_t>(0);
        for(int enabled:{0,1})for(int extent:{16,49,128})for(int contractions:{1,2,3,8,32,40,64}) {
            lookup=bool(enabled);dut.p_lut__enable.set<uint32_t>(enabled);
            columns=extent;pitch=(columns+15)/16*16;bands=contractions;
            dut.p_total__rows.set<uint32_t>(columns);dut.p_row__pitch.set<uint32_t>(pitch);
            dut.p_reduction__cols.set<uint32_t>(bands*16);dut.p_weight__cols.set<uint32_t>(16);
            for(block=0;block<3;++block) {
                valid_rows=block==2 ? 5 : 16;seen.assign(size_t(columns)*valid_rows,false);
                dut.p_weight__rows.set<uint32_t>(valid_rows);
                dut.p_committed__k__blk__idx.set<uint32_t>(block);
                monitor=true;
                for(band=0;band<bands;++band) {
                    dut.p_committed__n__blk__idx.set<uint32_t>(band);
                    dut.p_committed__first__k__blk.set<uint32_t>(band==0);
                    dut.p_a__en__last.set<uint32_t>(1);tick();
                    dut.p_a__en__last.set<uint32_t>(0);
                    int cycle=0;
                    do {
                        for(int lane=0;lane<16;++lane) {
                            const int col=cycle-lane;
                            dut.p_p__in.data[lane]=uint32_t(col>=0 && col<columns ? partial(band,block,lane,col):0);
                        }
                        tick();++cycle;
                        if(cycle>4096)throw std::runtime_error("Drain timeout");
                    }while(dut.p_busy.get<uint32_t>());
                    tick();tick();
                }
                monitor=false;
                if(std::find(seen.begin(),seen.end(),false)!=seen.end())throw std::runtime_error("Missing result");
                ++cases;
            }
        }
        std::cout<<"PASS native RTL CXXRTL: "<<cases<<" cases, "<<checks
                 <<" checked outputs; K up to1024, cancellation, tails, signed rounding, saturation\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
