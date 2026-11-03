# Run with quartus_sta -t this-script /absolute/path/to/adder_prototype
project_open [lindex $quartus(args) 0]
create_timing_netlist -model slow
read_sdc
update_timing_netlist
report_clocks -file clocks-reviewed.rpt
report_timing -setup -npaths 8 -to_clock [get_clocks {*pll_0*}] -detail full_path -file core-setup-reviewed.rpt
delete_timing_netlist
project_close
