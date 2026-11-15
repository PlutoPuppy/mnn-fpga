#!/usr/bin/env python3
"""Create standalone timing figures from the measured quartet experiment."""
import argparse
import json
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=pathlib.Path)
    a=p.parse_args();root=a.directory
    s=json.loads((root/'summary.json').read_text())
    rows=json.loads((root/'results.json').read_text())
    zeros=s.get('input_mode')=='zero_fp32'
    phase=lambda k:s['stages_ms'][k]['mean']
    q=s['selected_qk'];cpu=q['cpu']['mean'];dma=q['dma']['mean']
    groups=[('Input scan + quantization',phase('peak_scan')+phase('quantize'),'#3974b9'),
            ('Packing + workspace + scaling',sum(phase(k) for k in ('workspace','pack_pad','scale_bound')),'#76a6d2'),
            ('Input staging + DMA',phase('upload'),'#b3cce5'),
            ('FPGA execution + polling',phase('completion_wait'),'#19876d'),
            ('Output DMA + CPU copy',phase('download'),'#e3ad55'),
            ('Reconstruct + convert FP32',phase('dequantize')+phase('output_convert'),'#d1773e')]
    other=dma-sum(v for _,v,_ in groups)
    groups.append(('Registers, setup, logging, residual',other,'#8c839b'))
    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(16,5.7),gridspec_kw={'width_ratios':[1.2,1.1,1.3]})
    ax1.barh(1,cpu,color='#555d68',height=.45)
    left=0
    for label,value,color in groups:
        ax1.barh(0,value,left=left,height=.45,color=color,label=label);left+=value
    ax1.set_yticks([0,1],['DMA QK path','CPU QK']);ax1.set_xlabel('Milliseconds per selected QK operation')
    ax1.text(cpu+.3,1,'%.2f ms'%cpu,va='center');ax1.text(dma+.3,0,'%.2f ms'%dma,va='center')
    ax1.set_xlim(0,dma*1.18);ax1.set_title('Where the QK time goes')
    handles,labels=ax1.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.06,.01),ncol=2,frameon=False,fontsize=9)
    names=['CPU baseline','DMA offload'];means=[s['controls']['inference_ms'][m]['mean'] for m in ('cpu','dma')]
    bars=ax2.bar(names,means,color=['#555d68','#3974b9'],width=.55)
    ax2.set_ylabel('Milliseconds per full inference');ax2.set_ylim(0,max(means)*1.2)
    for b,v in zip(bars,means):ax2.text(b.get_x()+b.get_width()/2,v+15,'%.2f ms'%v,ha='center')
    ax2.set_title('Full model, detailed timers disabled\n%+.2f%% latency'%s['controls']['inference_ms']['latency_increase_pct'])
    delta=[r['runs']['dma_control']['inference_ms']-r['runs']['cpu_control']['inference_ms'] for r in rows]
    ax3.bar(range(1,21),delta,color=['#3974b9' if v>=0 else '#19876d' for v in delta])
    ax3.axhline(0,color='#555d68',linewidth=.8);ax3.axhline(sum(delta)/20,color='#d1773e',linestyle='--',label='Mean added latency')
    ax3.set_xticks([1,5,10,15,20]);ax3.set_xlabel('Zero-input repetition' if zeros else 'Random input index');ax3.set_ylabel('DMA minus CPU inference time (ms)')
    ax3.set_title('All 20 zero-input repetitions' if zeros else 'All 20 random inputs');ax3.legend(frameon=False,fontsize=8)
    for ax in (ax1,ax2,ax3):
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x' if ax==ax1 else 'y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('TinyViT first QK offload: 90 MHz, real DMA, 2-thread HPS CPU'+(' / all-zero model input' if zeros else ''),fontsize=14)
    fig.subplots_adjust(left=.09,right=.98,top=.79,bottom=.4,wspace=.6)
    for ext in ('png','svg'):fig.savefig(root/('timing-diagnosis.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
