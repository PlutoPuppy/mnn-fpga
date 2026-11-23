# Board reference clock. Derive PLL clocks from the regenerated hardware,
# rather than the upstream exported snapshot with fixed divider values.
create_clock -name c50 -period 20.000 [get_ports {CLOCK_50}]
derive_pll_clocks
derive_clock_uncertainty
