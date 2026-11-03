CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Werror
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Werror
CPPFLAGS += -Isw/include -Isw/tests
VERILATOR ?= verilator
BUILD_DIR ?= build/native
RTL = rtl/systolic_pe.sv rtl/systolic_array.sv rtl/systolic_mmio.sv
.PHONY: all sim lint clean test
test: $(BUILD_DIR)/tiling_test
	./$(BUILD_DIR)/tiling_test
$(BUILD_DIR)/tiling_test: sw/tests/tiling_test.cpp sw/src/fpga_gemm.cpp sw/src/matmul_soc.cpp sw/include/fpga_gemm.hpp sw/include/matmul_soc.hpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) sw/tests/tiling_test.cpp sw/src/fpga_gemm.cpp sw/src/matmul_soc.cpp -o $@
all: $(BUILD_DIR)/gemm_smoke_test
$(BUILD_DIR)/socfpga_bridge_control: sw/src/socfpga_bridge_control.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $< -o $@
$(BUILD_DIR)/matmul_soc_smoke: sw/src/matmul_soc_smoke.cpp sw/src/matmul_soc.cpp sw/src/matmul_soc_uio.cpp sw/include/matmul_soc.hpp sw/include/matmul_soc_uio.hpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) sw/src/matmul_soc_smoke.cpp sw/src/matmul_soc.cpp sw/src/matmul_soc_uio.cpp -o $@
build:
	mkdir -p build
$(BUILD_DIR)/gemm_smoke_test: sw/src/gemm_smoke_test.cpp sw/src/fpga_gemm.cpp sw/src/uio.cpp sw/include/fpga_gemm.hpp sw/include/uio.hpp sw/tests/test_vectors.hpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) sw/src/gemm_smoke_test.cpp sw/src/fpga_gemm.cpp sw/src/uio.cpp -o $@
sim: | build
	$(VERILATOR) --cc --exe --build -j 2 --top-module systolic_mmio --Mdir $(CURDIR)/build/obj_dir -Wall $(RTL) $(CURDIR)/sim/mmio_test.cpp $(CURDIR)/sw/src/fpga_gemm.cpp -CFLAGS "-std=c++17 -I$(CURDIR)/sw/include -I$(CURDIR)/sw/tests" -o mmio_test
	./build/obj_dir/mmio_test
lint:
	$(VERILATOR) --lint-only -Wall --top-module systolic_mmio $(RTL)
clean:
	$(RM) -r build
