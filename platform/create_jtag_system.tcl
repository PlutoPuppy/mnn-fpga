package require -exact qsys 23.1
create_system gemm_jtag
set_project_property DEVICE_FAMILY "Cyclone V"
set_project_property DEVICE 5CSEMA5F31C6
add_instance clock altera_clock_bridge
set_instance_parameter_value clock EXPLICIT_CLOCK_RATE 50000000
add_instance reset altera_reset_bridge
set_instance_parameter_value reset ACTIVE_LOW_RESET 1
set_instance_parameter_value reset SYNCHRONOUS_EDGES deassert
add_instance jtag altera_jtag_avalon_master
add_instance gemm systolic_mmio 1.0
add_connection clock.out_clk reset.clk
add_connection clock.out_clk jtag.clk
add_connection clock.out_clk gemm.clock
add_connection reset.out_reset jtag.clk_reset
add_connection reset.out_reset gemm.reset
add_connection jtag.master gemm.avs
set_connection_parameter_value jtag.master/gemm.avs baseAddress 0x00000000
add_interface clock clock end
set_interface_property clock EXPORT_OF clock.in_clk
add_interface reset reset end
set_interface_property reset EXPORT_OF reset.in_reset
save_system gemm_jtag.qsys
