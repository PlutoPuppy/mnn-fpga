# Quartus Prime Lite 23.1std / Cyclone V. Open in Platform Designer to validate.
package require -exact qsys 23.1
set_module_property NAME systolic_mmio
set_module_property VERSION 1.0
set_module_property DISPLAY_NAME "16x16 INT8 systolic GEMM"
set_module_property GROUP "MNN FPGA"
set_module_property INSTANTIATE_IN_SYSTEM_MODULE true
set_module_property EDITABLE true
set_module_property OPAQUE_ADDRESS_MAP true

add_fileset QUARTUS_SYNTH QUARTUS_SYNTH "" ""
set_fileset_property QUARTUS_SYNTH TOP_LEVEL systolic_mmio
add_fileset_file systolic_pe.sv SYSTEM_VERILOG PATH ../rtl/systolic_pe.sv
add_fileset_file systolic_array.sv SYSTEM_VERILOG PATH ../rtl/systolic_array.sv
add_fileset_file systolic_mmio.sv SYSTEM_VERILOG PATH ../rtl/systolic_mmio.sv TOP_LEVEL_FILE

add_interface clock clock end
add_interface_port clock clk clk Input 1
add_interface reset reset end
set_interface_property reset associatedClock clock
set_interface_property reset synchronousEdges BOTH
add_interface_port reset reset reset Input 1

add_interface avs avalon end
set_interface_property avs associatedClock clock
set_interface_property avs associatedReset reset
set_interface_property avs addressUnits WORDS
set_interface_property avs bitsPerSymbol 8
set_interface_property avs readLatency 1
set_interface_property avs readWaitTime 0
set_interface_property avs writeWaitTime 0
set_interface_property avs setupTime 0
set_interface_property avs holdTime 0
set_interface_property avs timingUnits Cycles
set_interface_property avs maximumPendingReadTransactions 0
add_interface_port avs avs_address address Input 10
add_interface_port avs avs_read read Input 1
add_interface_port avs avs_write write Input 1
add_interface_port avs avs_writedata writedata Input 32
add_interface_port avs avs_byteenable byteenable Input 4
add_interface_port avs avs_readdata readdata Output 32
add_interface_port avs avs_waitrequest waitrequest Output 1
