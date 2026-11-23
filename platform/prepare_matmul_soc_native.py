#!/usr/bin/env python3
"""Create an isolated 90-MHz native W8A8 Quartus project. Never programs hardware.

The native ABI accumulates the full reduction in INT32, then applies one
per-channel Q24 multiplier and Q16 bias. It is intentionally distinct from
the older per-band INT8 ABI; its capability word is NI82 at offset0x90,
or NI83 with the optional 32-KiB A operand bank.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import copy
import re


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Unsupported upstream source pattern: ' + old[:90])
    return text.replace(old, new, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--friend-root', type=pathlib.Path, required=True)
    p.add_argument('--output', type=pathlib.Path, required=True)
    p.add_argument('--extended-a', action='store_true', help='NI83: 32 KiB A bank, K up to2048, B write base0x9000')
    a = p.parse_args()
    source, output = a.friend_root.resolve(), a.output.resolve()
    repo = pathlib.Path(__file__).resolve().parent.parent
    if not (source/'adder_prototype.qpf').is_file() or output.exists() or source == output:
        p.error('Use an existing upstream project and a new isolated output directory')
    output.mkdir(parents=True)
    for item in source.iterdir():
        if item.is_file() and item.suffix.lower() in ('.v', '.qip', '.qsf', '.qpf', '.qsys', '.sdc', '.mif', '.hex', '.tcl'):
            shutil.copy2(item, output/item.name)
    shutil.copytree(source/'matmul_accel', output/'matmul_accel')
    # Qsys regeneration creates fresh generated synthesis files.
    top = output/'matmul_accel/accel_top.sv'
    text = top.read_text(encoding='utf-8')
    text = replace_once(text, 'logic [AVS_DATA_WIDTH-1:0] ctrl_rdata;', '''wire matmul_busy;
    logic [AVS_DATA_WIDTH-1:0] ctrl_rdata_legacy;
    wire [AVS_DATA_WIDTH-1:0] ctrl_rdata =
        avs_address == 32'h90 ? {{(AVS_DATA_WIDTH-32){1'b0}},32'h4e493832} :
        avs_address == 32'ha0 ? {{(AVS_DATA_WIDTH-32){1'b0}},32'h01181080} : ctrl_rdata_legacy;
    wire native_cfg_write = avs_write_accepted && !matmul_busy &&
        avs_address >= 32'h19000 && avs_address < 32'h19400;
    wire [6:0] native_cfg_index = (avs_address-32'h19000)>>3;
    wire native_lut_write = avs_write_accepted && !matmul_busy &&
        avs_address >= 32'h19400 && avs_address < 32'h19500;
    reg [1:0] native_mode;
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) native_mode <= 0;
        else if (avs_write_accepted && !matmul_busy && avs_address==32'hb0)
            native_mode <= avs_writedata[1:0];
    end''')
    text = replace_once(text, '.rdata  (ctrl_rdata)', '.rdata  (ctrl_rdata_legacy)')
    text = replace_once(text, 'wire        matmul_busy = weight_ctrl_busy || output_loader_busy;',
                        'assign matmul_busy = weight_ctrl_busy || output_loader_busy;')
    text = replace_once(text, 'output_loader #(', 'output_loader_native #(')
    text = replace_once(text, '.output_scale   (output_scale),', '''.reduction_cols (weight_cols),
        .cfg_write (native_cfg_write), .cfg_index (native_cfg_index), .cfg_data (avs_writedata),
        .lut_write (native_lut_write), .lut_index (avs_address[7:4]),
        .lut_data (avs_writedata), .lut_enable (native_mode[0]),''')
    text = replace_once(text, '.tile_row_blk   (tile_row_blk)', '.tile_row_blk   (native_mode[1] ? 16\'d0 : tile_row_blk)')
    text = replace_once(text, '.tile_col_off   (tile_col_off)', '.tile_col_off   (native_mode[1] ? 12\'d3840 : tile_col_off)')
    # Input B writes may overlap the READ-ONLY output aperture: Avalon read
    # and write decoders are separate. This keeps every existing address.
    text = replace_once(text, "IBUF_BASE = 32'h0000_5000, IBUF_SIZE = 32'h0000_4000",
                        "IBUF_BASE = 32'h0000_5000, IBUF_SIZE = 32'h0001_0000")
    text = replace_once(text, 'localparam int IBUF_BANK_ADDR_WIDTH = 10;', 'localparam int IBUF_BANK_ADDR_WIDTH = 12;')
    text = replace_once(text, 'logic [9:0]  input_max_addr;', '''logic [9:0]  input_max_addr;
    wire [IBUF_BANK_ADDR_WIDTH-1:0] input_extent = input_max_addr;''')
    if text.count('input_max_addr[IBUF_BANK_ADDR_WIDTH-1:0]') != 3:
        raise ValueError('Unsupported input extent wiring')
    text = text.replace('input_max_addr[IBUF_BANK_ADDR_WIDTH-1:0]', 'input_extent')
    if a.extended_a:
        text = replace_once(text, "32'h4e493832", "32'h4e493833")
        text = replace_once(text, "32'h01181080", "32'h03181080")
        text = replace_once(text, "WBUF_BASE = 32'h0000_1000, WBUF_SIZE = 32'h0000_4000",
                            "WBUF_BASE = 32'h0000_1000, WBUF_SIZE = 32'h0000_8000")
        text = replace_once(text, "IBUF_BASE = 32'h0000_5000, IBUF_SIZE = 32'h0001_0000",
                            "IBUF_BASE = 32'h0000_9000, IBUF_SIZE = 32'h0001_0000")
        text = replace_once(text, 'localparam int WBUF_BANK_ADDR_WIDTH = 10;', 'localparam int WBUF_BANK_ADDR_WIDTH = 11;')
    top.write_text(text, encoding='utf-8')
    # The old 128-bit x64 write port already uses four wide RAM blocks per
    # bank. x256 fills their depth while retaining the same read latency.
    ibuf = output/'input_buffer.v'
    text = ibuf.read_text(encoding='utf-8')
    for old, new in [(r'(\binput\s+)\[9:0\](\s+rdaddress;)',r'\g<1>[11:0]\2'),
                     (r'(\binput\s+)\[5:0\](\s+wraddress;)',r'\g<1>[7:0]\2'),
                     (r'(\.numwords_a\s*=\s*)64\b',r'\g<1>256'),
                     (r'(\.numwords_b\s*=\s*)1024\b',r'\g<1>4096'),
                     (r'(\.widthad_a\s*=\s*)6\b',r'\g<1>8'),
                     (r'(\.widthad_b\s*=\s*)10\b',r'\g<1>12')]:
        text, count = re.subn(old, new, text)
        if count != 1:
            raise ValueError('Unsupported input buffer IP pattern: '+old)
    ibuf.write_text(text, encoding='utf-8')
    if a.extended_a:
        wbuf = output/'weight_buffer.v'
        text = wbuf.read_text(encoding='utf-8')
        for old, new in [(r'(\binput\s+)\[9:0\](\s+rdaddress;)',r'\g<1>[10:0]\2'),
                         (r'(\binput\s+)\[5:0\](\s+wraddress;)',r'\g<1>[6:0]\2'),
                         (r'(\.numwords_a\s*=\s*)64\b',r'\g<1>128'),
                         (r'(\.numwords_b\s*=\s*)1024\b',r'\g<1>2048'),
                         (r'(\.widthad_a\s*=\s*)6\b',r'\g<1>7'),
                         (r'(\.widthad_b\s*=\s*)10\b',r'\g<1>11')]:
            text, count = re.subn(old,new,text)
            if count != 1:raise ValueError('Unsupported weight buffer IP pattern: '+old)
        wbuf.write_text(text,encoding='utf-8')
    shutil.copy2(repo/'rtl/matmul_soc/output_loader_native.sv', output/'matmul_accel/output_loader_native.sv')
    control = output/'matmul_accel/weight_ctrl.v'
    text = control.read_text(encoding='utf-8')
    old = '''if (is_last_k_blk) begin
                                    k_blk_idx <= 0;
                                    n_blk_idx <= n_blk_idx + 1'b1;
                                end
                                else begin
                                    k_blk_idx <= k_blk_idx + 1'b1;
                                end'''
    new = '''// Native ABI: finish all reduction-column bands for one
                                // output-row block before reusing its INT32 RAM.
                                if (is_last_n_blk) begin
                                    n_blk_idx <= 0;
                                    k_blk_idx <= k_blk_idx + 1'b1;
                                end
                                else begin
                                    n_blk_idx <= n_blk_idx + 1'b1;
                                end'''
    text = replace_once(text, old, new)
    control.write_text(text, encoding='utf-8')
    tcl = output/'matmul_accel_hw.tcl'
    text = tcl.read_text(encoding='utf-8')
    text = replace_once(text, 'add_fileset_file output_loader.v VERILOG PATH matmul_accel/output_loader.v',
                        'add_fileset_file output_loader_native.sv SYSTEM_VERILOG PATH matmul_accel/output_loader_native.sv')
    tcl.write_text(text, encoding='utf-8')
    qsf = output/'adder_prototype.qsf'
    text = qsf.read_text(encoding='utf-8')
    lines = [line for line in text.splitlines() if 'auto_signaltap_0' not in line and
             not any(key in line for key in ('USE_SIGNALTAP_FILE', 'SIGNALTAP_FILE', 'SLD_FILE db/stp'))]
    lines = [line.replace('ENABLE_SIGNALTAP ON', 'ENABLE_SIGNALTAP OFF') for line in lines]
    qsf.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    subprocess.run([sys.executable, str(repo/'platform/apply_matmul_soc_profile.py'), str(output/'adder_prototype.qsys'),
                    '--profile', str(repo/'platform/de1-soc-matmul-90mhz-profile.json')], check=True)
    qsys = output/'adder_prototype.qsys'
    tree = ET.parse(qsys)
    root = tree.getroot()
    # This IP emits one burst per descriptor. The matching native driver caps
    # HPS descriptors at 128*16=2048 bytes and local descriptors at16*16=256.
    # 256-entry HPS FIFOs hold two maximum bursts, with ample headroom.
    for name in ('dma_0', 'dma_1'):
        dma = root.find("module[@name='%s']" % name)
        dma.find("parameter[@name='fifoDepth']").set('value', '256')
        dma.find("parameter[@name='maxBurstSize']").set('value', '128')
    local_dma = copy.deepcopy(root.find("module[@name='dma_1']"))
    local_dma.set('name', 'dma_2')
    local_dma.find("parameter[@name='fifoDepth']").set('value', '64')
    local_dma.find("parameter[@name='maxBurstSize']").set('value', '16')
    root.append(local_dma)
    for connection in list(root.findall('connection')):
        if connection.get('end', '').startswith('dma_1.'):
            cloned = copy.deepcopy(connection)
            cloned.set('end', cloned.get('end').replace('dma_1.', 'dma_2.'))
            if cloned.get('kind') == 'avalon':
                cloned.find("parameter[@name='baseAddress']").set('value', '0x0040')
            root.append(cloned)
    for port in ('read_master', 'write_master'):
        connection = ET.SubElement(root, 'connection', kind='avalon', version='23.1',
                                   start='dma_2.'+port, end='matmul_accel_0.avalon_slave_0')
        for name, value in [('arbitrationPriority', '1'), ('baseAddress', '0x0000'), ('defaultConnection', 'false')]:
            ET.SubElement(connection, 'parameter', name=name, value=value)
    tree.write(qsys, encoding='UTF-8', xml_declaration=True)
    proof = dict(abi='NI82', capability=0x4e493832, capability_offset=0x90, parameter_offset=0x19000,
                 parameter_bytes=1024, max_output_columns=128, scale_fraction_bits=24, bias_fraction_bits=16,
                 partial_sum_bits=32, signal_tap=False, activation_lut_offset=0x19400,
                 mode_offset=0xb0, mode_lut_bit=0, mode_scratch_bit=1,
                 local_dma_csr_offset=0x40, scratch_output_offset=0x18000,
                 operand_a_bytes=16384, operand_b_bytes=65536, output_bytes=65536,
                 source=str(source),
                 sources={str(f.relative_to(output)): hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in (top, control, ibuf, output/'matmul_accel/output_loader_native.sv', qsys, qsf, output/'mnn_matmul_soc.sdc')
                          if f.is_file()})
    if a.extended_a:
        proof.update(abi='NI83',capability=0x4e493833,operand_a_bytes=32768,operand_b_offset=0x9000,max_reduction=2048)
        proof['sources']['weight_buffer.v']=hashlib.sha256((output/'weight_buffer.v').read_bytes()).hexdigest()
    (output/'native-abi.json').write_text(json.dumps(proof, indent=2)+'\n')
    print('Prepared native W8A8 project:', output)


if __name__ == '__main__':
    main()
