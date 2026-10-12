create_clock -name CLOCK_50 -period 20.000 [get_ports CLOCK_50]
derive_pll_clocks
derive_clock_uncertainty
# The pushbutton is asynchronous; the reset bridge synchronizes release.
set_false_path -from [get_ports {KEY[*]}]
# LEDs are static indicators, not a synchronous output interface.
set_false_path -to [get_ports {LEDR[*]}]
