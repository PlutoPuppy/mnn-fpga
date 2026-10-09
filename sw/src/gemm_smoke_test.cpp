#include "uio.hpp"
#include "test_vectors.hpp"
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: gemm_smoke_test /dev/uioN\n"; return 2;
    }
    try {
        fpga::Uio io(argv[1]);
        fpga::run_vectors(io);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n'; return 1;
    }
}
