#!/usr/bin/env python3
"""Clone generated NI83 synthesis inputs and wrap the three DMA CSR ports."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    src, dst = a.source.resolve(), a.output.resolve()
    if not (src / 'native-abi.json').is_file() or dst.exists() or src in dst.parents:
        p.error('Use a generated NI83 project and a new sibling output directory')
    if json.loads((src / 'native-abi.json').read_text())['abi'] != 'NI83':
        p.error('Expected NI83 synthesis inputs')
    dst.mkdir()
    for f in src.glob('*'):
        if f.is_file() and f.suffix in {'.qpf', '.qsf', '.qsys', '.v', '.qip', '.sdc', '.json', '.tcl'}:
            shutil.copy2(f, dst / f.name)
    for folder in ('adder_prototype', 'matmul_accel', 'hps_isw_handoff'):
        shutil.copytree(src / folder, dst / folder)
    qsf = dst / 'adder_prototype.qsf'
    control_names = ('dma_ctl_address', 'dma_ctl_chipselect', 'dma_ctl_write_n',
                     'dma_ctl_writedata', 'dma_ctl_readdata')
    repo = Path(__file__).resolve().parent.parent
    rtl = repo / 'rtl/matmul_soc/dma_burst_chain.v'
    shutil.copy2(rtl, dst / rtl.name)
    submodules = dst / 'adder_prototype/synthesis/submodules'
    generated = []
    for i, burst in enumerate((2048, 2048, 256)):
        module = 'adder_prototype_dma_' + str(i)
        file = submodules / (module + '.v')
        content = file.read_text()
        begin = content.index('module ' + module + ' (')
        pivot = content.index('reg              burst_read_waitrequest_s1;', begin)
        end = content.index('endmodule', pivot)
        # Keep the Qsys-facing port declarations and subordinate DMA modules.
        # Only redirect the main DMA's internal CSR signals to the sequencer.
        names = re.compile(r'(?<!\.)\b(' + '|'.join(control_names) + r')\b')
        core = names.sub(lambda match: 'child_' + match[0], content[pivot:end])
        declaration = re.compile(r'^reg\s+\[\s*31\s*:\s*0\s*\]\s+child_dma_ctl_readdata;', re.M)
        if len(declaration.findall(core)) != 1:
            raise ValueError('Unsupported DMA read-data declaration: ' + module)
        insert = f'''
wire [2:0] child_dma_ctl_address;
wire child_dma_ctl_chipselect, child_dma_ctl_write_n;
wire [31:0] child_dma_ctl_writedata;
dma_burst_chain #(.BURST_BYTES({burst})) burst_chain (
    .clk(clk), .reset_n(system_reset_n),
    .addr(dma_ctl_address), .cs(dma_ctl_chipselect), .wr_n(dma_ctl_write_n),
    .wdata(dma_ctl_writedata), .rdata(dma_ctl_readdata),
    .child_addr(child_dma_ctl_address), .child_cs(child_dma_ctl_chipselect),
    .child_wr_n(child_dma_ctl_write_n), .child_wdata(child_dma_ctl_writedata),
    .child_rdata(child_dma_ctl_readdata)
);
'''
        file.write_text(content[:pivot] + insert + core + content[end:])
        generated.append(file)
    with qsf.open('a') as f:
        f.write('\nset_global_assignment -name VERILOG_FILE dma_burst_chain.v\n')
    generated.extend([dst / rtl.name, qsf])
    (dst / 'dma-chain.json').write_text(json.dumps({
        'source_project': str(src), 'capability_register': 4, 'capability_word': '0x444d4331',
        'max_command_bytes': 65536, 'burst_bytes': [2048, 2048, 256],
        'sources': {f.relative_to(dst).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in generated}}, indent=2) + '\n')
    print('Prepared NI83 DMA-chain project:', dst)


if __name__ == '__main__':
    main()
