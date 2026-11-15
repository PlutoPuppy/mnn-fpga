# quartus_sta -t verify_matmul_clock.tcl PROJECT EXPECTED_MHZ
# Check the fitted clock, not just the requested Platform Designer parameter.
if {[llength $quartus(args)] != 2} {
    error "Expected PROJECT EXPECTED_MHZ"
}
set expected_mhz [lindex $quartus(args) 1]
if {![string is double -strict $expected_mhz] || $expected_mhz <= 0} {
    error "EXPECTED_MHZ must be positive"
}
project_open [lindex $quartus(args) 0]
create_timing_netlist -model slow
read_sdc
update_timing_netlist
set core_clocks [get_clocks {*pll_0*PLL_OUTPUT_COUNTER*divclk}]
set count 0
foreach_in_collection clock $core_clocks {
    incr count
    set period [get_clock_info -period $clock]
    set actual_mhz [expr {1000.0 / $period}]
    puts "CORE_CLOCK name=[get_clock_info -name $clock] period_ns=$period frequency_mhz=$actual_mhz"
    if {abs($actual_mhz - $expected_mhz) > 0.05} {
        error "Fitted core clock differs from expected $expected_mhz MHz"
    }
}
if {$count != 1} {
    error "Expected exactly one core output clock, found $count"
}
report_clocks -file fitted-clocks-reviewed.rpt
report_timing -setup -npaths 10 -to_clock $core_clocks -detail full_path -file fitted-core-setup-reviewed.rpt
report_timing -hold -npaths 10 -to_clock $core_clocks -detail full_path -file fitted-core-hold-reviewed.rpt
puts "CORE_CLOCK_VERIFIED expected_mhz=$expected_mhz"
delete_timing_netlist
project_close
