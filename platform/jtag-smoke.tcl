# Run with System Console --script=platform/jtag-smoke.tcl after programming.
# Addresses are BYTES here; the Platform Designer adapter converts to words.
proc gemm_read {master address} {
    return [expr {[lindex [master_read_32 $master $address 1] 0] & 0xffffffff}]
}
proc gemm_signed {word} {
    set word [expr {$word & 0xffffffff}]
    return [expr {$word >= 0x80000000 ? $word - 0x100000000 : $word}]
}
proc gemm_reference {a b} {
    set c {}
    for {set i 0} {$i < 16} {incr i} {
        for {set j 0} {$j < 16} {incr j} {
            set sum 0
            for {set k 0} {$k < 16} {incr k} {
                incr sum [expr {[lindex $a [expr {16*$i+$k}]] * [lindex $b [expr {16*$k+$j}]]}]
            }
            lappend c $sum
        }
    }
    return $c
}
proc gemm_check_tile {master name a b} {
    if {[llength $a] != 256 || [llength $b] != 256} { error "Tile must contain 256 elements" }
    set packed_a {}; set packed_b {}
    foreach value $a {
        if {![string is integer -strict $value] || $value < -128 || $value > 127} { error "Invalid INT8 A" }
        lappend packed_a [expr {$value & 255}]
    }
    foreach value $b {
        if {![string is integer -strict $value] || $value < -128 || $value > 127} { error "Invalid INT8 B" }
        lappend packed_b [expr {$value & 255}]
    }
    master_write_32 $master 0x008 {4}
    master_write_32 $master 0x400 $packed_a
    master_write_32 $master 0x800 $packed_b
    # Verify the transfer before starting computation.
    foreach address {0x400 0x800} expected [list $a $b] {
        set uploaded [master_read_32 $master $address 256]
        if {[llength $uploaded] != 256} { error "Short operand read" }
        for {set i 0} {$i < 256} {incr i} {
            if {[gemm_signed [lindex $uploaded $i]] != [lindex $expected $i]} {
                error "$name operand readback failed at $i"
            }
        }
    }
    master_write_32 $master 0x008 {1}
    set deadline [expr {[clock milliseconds] + 5000}]
    while {1} {
        set status [gemm_read $master 0x00c]
        if {$status & 4} { error "$name accelerator error, status=$status" }
        if {($status & 3) == 2} { break }
        if {[clock milliseconds] >= $deadline} {
            master_write_32 $master 0x008 {4}
            error "$name accelerator timeout"
        }
        after 1
    }
    set actual [master_read_32 $master 0xc00 256]
    set expected [gemm_reference $a $b]
    if {[llength $actual] != 256} { error "Short result read" }
    for {set i 0} {$i < 256} {incr i} {
        set value [gemm_signed [lindex $actual $i]]
        if {$value != [lindex $expected $i]} {
            error "$name C\[[expr {$i/16}], [expr {$i%16}]\]: got $value expected [lindex $expected $i]"
        }
    }
    set cycles [gemm_read $master 0x010]
    if {$cycles != 384} { error "$name unexpected cycle count $cycles" }
    master_write_32 $master 0x008 {2}
    puts "PASS $name: 256 INT32 outputs, $cycles fabric clocks"
}
proc gemm_smoke {master} {
    foreach address {0x000 0x004 0x018} expected {0x53413136 0x00010000 0x20101008} {
        if {[gemm_read $master $address] != $expected} { error "Wrong GEMM ID/version/format at $address" }
    }
    master_write_32 $master 0x014 {0xa55a1234}
    if {[gemm_read $master 0x014] != 0xa55a1234} { error "Scratch readback failed" }
    set zero [lrepeat 256 0]
    set a $zero; set b $zero
    foreach index {0 1 16 17} av {1 2 3 4} bv {5 6 7 8} {
        lset a $index $av; lset b $index $bv
    }
    gemm_check_tile $master embedded_2x2 $a $b
    gemm_check_tile $master zero $zero $a
    set identity $zero; set signed {}
    for {set i 0} {$i < 256} {incr i} {
        if {$i/16 == $i%16} { lset identity $i 1 }
        lappend signed [expr {($i*73+19)%256-128}]
    }
    gemm_check_tile $master left_identity $identity $signed
    gemm_check_tile $master right_identity $signed $identity
    gemm_check_tile $master int8_min [lrepeat 256 -128] [lrepeat 256 -128]
    gemm_check_tile $master opposite_extremes [lrepeat 256 -128] [lrepeat 256 127]
    set a $zero; set b $zero
    lset a 240 -128; lset b 15 127
    gemm_check_tile $master far_corner_k0 $a $b
    for {set seed 1} {$seed <= 5} {incr seed} {
        set a {}; set b {}; set random $seed
        for {set i 0} {$i < 256} {incr i} {
            set random [expr {(1664525*$random+1013904223)&0xffffffff}]
            lappend a [expr {(($random >> 16)&255)-128}]
            set random [expr {(1664525*$random+1013904223)&0xffffffff}]
            lappend b [expr {(($random >> 16)&255)-128}]
        }
        gemm_check_tile $master signed_random_$seed $a $b
    }
    puts "PASS: 12 FPGA matrix multiplications; all 3072 outputs match CPU reference"
}
if {![info exists ::GEMM_TEST_LIBRARY_ONLY]} {
    set masters [get_service_paths master]
    if {[info exists ::env(GEMM_MASTER_INDEX)]} {
        set index $::env(GEMM_MASTER_INDEX)
        if {![string is integer -strict $index] || $index < 0 || $index >= [llength $masters]} {
            error "GEMM_MASTER_INDEX is out of range: $masters"
        }
    } elseif {[llength $masters] == 1} {
        set index 0
    } else {
        error "Expected one JTAG master; found [llength $masters]: $masters. Select GEMM_MASTER_INDEX explicitly."
    }
    set master [claim_service master [lindex $masters $index] mnn-fpga-gemm]
    set result [catch {gemm_smoke $master} message options]
    if {$result} { catch {master_write_32 $master 0x008 {4}} }
    close_service master $master
    if {$result} { return -options $options $message }
}
