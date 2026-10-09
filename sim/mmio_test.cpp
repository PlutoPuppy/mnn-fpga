#include "Vsystolic_mmio.h"
#include "verilated.h"
#include "fpga_gemm.hpp"
#include "test_vectors.hpp"
#include <limits>
#include <iostream>
#include <stdexcept>

class SimIo final : public fpga::RegisterIo {
public:
    Vsystolic_mmio dut;
    SimIo() {
        dut.clk=0; dut.reset=1; dut.avs_read=0; dut.avs_write=0;
        dut.avs_address=0; dut.avs_writedata=0; dut.avs_byteenable=15;
        tick(); tick(); dut.reset=0; tick();
    }
    void tick() { dut.clk=0; dut.eval(); dut.clk=1; dut.eval(); dut.clk=0; dut.eval(); }
    uint32_t read(unsigned off) override {
        dut.avs_address=off/4; dut.avs_read=1; tick();
        auto v=dut.avs_readdata; dut.avs_read=0; return v;
    }
    void write(unsigned off, uint32_t v) override { write_mask(off,v,15); }
    void write_mask(unsigned off, uint32_t v, unsigned mask) {
        dut.avs_address=off/4; dut.avs_writedata=v; dut.avs_byteenable=mask;
        dut.avs_write=1; tick(); dut.avs_write=0; dut.avs_byteenable=15;
    }
};
static void require(bool v, const char* message) { if (!v) throw std::runtime_error(message); }
class StuckIo final : public fpga::RegisterIo {
public:
    bool started=false, aborted=false, fail=false;
    uint32_t read(unsigned o) override {
        using namespace fpga;
        if(o==reg::Id) return reg::Magic;
        if(o==reg::Version) return reg::Abi;
        if(o==reg::Format) return reg::FormatValue;
        if(o==reg::Status) return started ? (fail ? reg::Error : reg::Busy) : 0;
        return 0;
    }
    void write(unsigned o, uint32_t v) override {
        if(o==fpga::reg::Control && v==fpga::reg::Start) started=true;
        if(o==fpga::reg::Control && v==fpga::reg::Abort) aborted=true;
    }
};
int main(int argc, char** argv) {
    Verilated::commandArgs(argc,argv);
    try {
        using namespace fpga;
        SimIo io;
        run_vectors(io);
        io.write(reg::Scratch,0x11223344);
        io.write_mask(reg::Scratch,0xaabbccdd,5);
        require(io.read(reg::Scratch)==0x11bb33dd,"Scratch byte enables");
        io.write(reg::A,42);
        io.write_mask(reg::A,99,2);
        require(io.read(reg::A)==42,"Input byte enable must select byte zero");
        io.write_mask(reg::Control,reg::Start,2);
        require(!(io.read(reg::Status)&reg::Busy),"Masked START took effect");
        io.write(reg::Control,reg::Start);
        require(io.read(reg::Status)&reg::Busy,"START did not set busy");
        io.write(reg::A,99);
        require(io.read(reg::A)==42,"Busy input write modified buffer");
        require(io.read(reg::Status)&reg::Error,"Busy write not rejected");
        io.write(reg::Control,reg::Ack);
        io.write(reg::Control,reg::Start);
        require(io.read(reg::Status)&reg::Error,"Busy START not rejected");
        io.write(reg::Control,reg::Abort);
        require(io.read(reg::Status)==0,"Abort did not return idle");
        io.write(reg::C,123);
        require(io.read(reg::Status)&reg::Error,"Output write not rejected");
        io.write(reg::Control,reg::Ack);
        require(io.read(reg::Status)==0,"ACK did not clear status");
        // Reset during RUN and recover using the same software submission path.
        io.write(reg::Control,reg::Start);
        for(int i=0;i<12;++i) io.tick();
        io.dut.reset=1; io.tick(); io.dut.reset=0; io.tick();
        require(io.read(reg::Status)==0,"Reset did not return idle");
        Input a{},b{}; a.fill(-3); b.fill(7); Output c{};
        gemm(io,a,b,c); require(c==reference(a,b),"Recovery GEMM mismatch");
        std::array<float,Elements> fa{},fb{},fc{};
        fa.fill(-3); fb.fill(7);
        require(exact_float_tile(io,fa.data(),fb.data(),fc.data()),"Exact float tile rejected");
        for(auto x:fc) require(x==-336.0f,"Exact float tile mismatch");
        for(float bad : {0.5f,128.0f,-129.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
            fa[0]=bad;
            require(!exact_float_tile(io,fa.data(),fb.data(),fc.data()),"Unsupported float accepted");
        }
        for(bool error : {false,true}) {
            StuckIo stuck; stuck.fail=error; bool threw=false;
            try { gemm(stuck,a,b,c,std::chrono::milliseconds(1)); }
            catch(const std::runtime_error&) { threw=true; }
            require(threw && stuck.aborted,"Timeout/error did not abort");
        }
        std::cout << "PASS protocol, reset/recovery, float eligibility, timeout/error handling\n";
    } catch(const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
