#!/usr/bin/env python3
"""Export end-to-end and disjoint offload-phase plots for NI82/NI83 evidence."""
import argparse
import json
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=pathlib.Path);root=p.parse_args().directory
    summary=json.loads((root/'native-summary.json').read_text());modes=summary['metadata']['modes']
    labels={'fp32':'FP32 CPU','cpu':'Corrected CPU','all':'FPGA: 30 linears','pairs':'FPGA: 30 + resident MLPs',
            'selective':'FPGA: selected','selective_pairs':'FPGA: selected + resident MLPs',
            'transformer_cpu':'40 transformer linears: CPU','transformer_fpga':'40 transformer linears + 10 pairs',
            'static_cpu':'51 static linears: CPU','static_fpga':'51 static linears + 10 pairs',
            'static_unfused':'51 static linears: unfused FPGA',
            'fast':'FPGA + faster polling','optimized':'FPGA + polling/attention/GELU',
            'cpu_optimized':'FP32 CPU + attention/GELU'}
    short={'fp32':'FP32\nCPU','cpu':'Mixed W8A8\nCPU','all':'FPGA\n30 linears',
           'pairs':'FPGA\n30 + 4 pairs','selective':'FPGA\nselected linears',
           'selective_pairs':'FPGA\n22 + resident MLPs',
           'transformer_cpu':'40 linears\nCPU','transformer_fpga':'40 linears\nFPGA\n10 pairs',
           'static_cpu':'51 linears\nCPU','static_fpga':'51 linears\nFPGA\n10 pairs',
           'static_unfused':'51 linears\nFPGA\nunfused',
           'fast':'FPGA\nfast polling','optimized':'FPGA\nCPU kernels',
           'cpu_optimized':'FP32 CPU\nCPU kernels',
           'previous':'FPGA\nprevious kernels','no_park':'FPGA\nnew kernels'}
    if len(modes)>6:
        short.update(cpu='Validated\nmixed CPU',pairs='30 linears\nFPGA\n4 pairs')
    configs=summary['metadata'].get('configurations',{})
    labels.update(previous='FPGA: previous CPU kernels',no_park='FPGA: new CPU kernels')
    if '--park-fpga-workers' in configs.get('optimized',{}).get('flags',[]):
        labels['optimized']='FPGA: new CPU kernels + worker parking'
        short['optimized']='FPGA\nnew + parking'
    if configs.get('optimized', {}).get('environment', {}).get('MNN_MATMUL_SOC_REUSE_A') == '1':
        labels.update(baseline='FPGA: v25 transport', optimized='FPGA: tile reuse + live rows',
                      poll_batch='FPGA: same schedule + batch polling')
        short.update(baseline='FPGA\nv25 transport', optimized='FPGA\ntile reuse + live rows',
                     poll_batch='FPGA\n+ batch polling')
    hardware=[m for m in modes if configs.get(m,{}).get('offload',m not in ('cpu','fp32'))]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(16,6) if len(modes)>6 else (13.5,5.5),layout='constrained')
    values=[summary['configurations'][m]['inference_ms']['mean'] for m in modes]
    errors=[summary['configurations'][m]['inference_ms']['sd'] for m in modes]
    bars=ax.bar(range(len(modes)),values,yerr=errors,capsize=4,color=['#287f79' if m in hardware else '#546779' for m in modes])
    ax.set_xticks(range(len(modes)),[short.get(m,m) for m in modes],fontsize=8)
    for bar,value,error in zip(bars,values,errors):ax.text(bar.get_x()+bar.get_width()/2,value+error+12,'%.1f'%value,ha='center',fontsize=9)
    ax.set_ylim(0,max(v+e for v,e in zip(values,errors))*1.12);ax.set_ylabel('Full inference (ms), mean ± SD')
    ax.set_title('%d paired images; detailed timers disabled'%summary['metadata']['inputs']);ax.spines[['top','right']].set_visible(False)
    if 'fp32' in modes:
        target=summary['configurations']['fp32']['inference_ms']['mean']/1.5
        ax.axhline(target,color='#9c4848',linestyle='--',linewidth=1)
        ax.text(-.4,target+12,'1.5x target: %.1f ms'%target,color='#9c4848',fontsize=8)
    groups=[('Input transpose/copy + DMA',['upload'],'#d7b25f'),('Register/parameter setup',['configure'],'#b6cddd'),
            ('FPGA execution + polling',['completion_wait','launch'],'#287f79'),('FPGA-local feedback',['local_feedback'],'#a48ec2'),
            ('Output DMA + transpose/copy',['download','output_layout'],'#d9c9a7'),('Host allocations',['layout_allocate','workspace'],'#c47b56')]
    for y,m in enumerate(hardware):
        start=0
        for label,keys,color in groups:
            value=sum(summary['phases'][m].get(k,{}).get('mean',0) for k in keys)
            bx.barh(y,value,left=start,color=color,label=label if y==0 else None);start+=value
        residual=summary['phases'][m]['adapter_total']['mean']-start
        if residual < -.01:raise ValueError('Plotted adapter phases overlap')
        bx.barh(y,max(0,residual),left=start,color='#a6a6a6',label='Other adapter control' if y==0 else None)
        start+=max(0,residual)
        bx.text(start+1,y,'%.2f ms'%start,va='center',fontsize=9)
    bx.set_yticks(range(len(hardware)),[labels.get(m,m) for m in hardware],fontsize=9)
    bx.set_xlabel('Offload adapter phases (ms), separate profiled runs')
    bx.set_title('HPS transfers and FPGA-local transfers shown separately')
    bx.spines[['top','right']].set_visible(False);bx.legend(loc='upper left',bbox_to_anchor=(0,-.16),fontsize=8,ncol=2)
    title='TinyViT / DE1-SoC / 90 MHz '+str(summary['metadata'].get('abi','NI82'))
    if any(m in modes for m in ('transformer_cpu','static_cpu')):
        title+='\nExpanded 40/51-layer quantization policies fail the CPU accuracy gate'
    fig.suptitle(title,fontsize=13)
    for ext in ('png','svg'):fig.savefig(root/('native-performance.'+ext),dpi=170,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)


if __name__=='__main__':main()
