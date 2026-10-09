CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Werror
CPPFLAGS += -Isw/include -Isw/tests
VERILATOR ?= verilator
BUILD_DIR ?= build/native
RTL = rtl/systolic_pe.sv rtl/systolic_array.sv rtl/systolic_mmio.sv
.PHONY: all sim lint clean
all: $(BUILD_DIR)/gemm_smoke_test
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
