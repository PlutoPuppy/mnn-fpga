#!/usr/bin/env python3
"""Export timing plots for the measured native W8A8 fc1 experiment."""
import argparse,json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=pathlib.Path);root=p.parse_args().directory
    s=json.loads((root/'summary.json').read_text());rows=json.loads((root/'results.json').read_text())
    meta=json.loads((root/'metadata.json').read_text())
    repaired=pathlib.Path(meta['models']['full_cpu']['file']).name=='tiny-vit-mixed-w8a8-accuracy.mnn'
    broad_label='Corrected mixed W8A8' if repaired else 'Broad W8A8'
    families=('fp32','full_cpu','full_fpga','hybrid_cpu','hybrid_fpga')
    labels=('FP32 CPU','Broad W8A8\nCPU','Broad W8A8\n+ FPGA fc1','fc1-only W8A8\nCPU','fc1-only W8A8\n+ FPGA fc1')
    if repaired:labels=('FP32 CPU','Corrected mixed\nW8A8 CPU','Corrected mixed\nW8A8 + FPGA','fc1-only W8A8\nCPU','fc1-only W8A8\n+ FPGA fc1')
    colors=('#555e69','#8fa9c6','#397da6','#9bc7bd','#287f79')
    fig,axes=plt.subplots(2,2,figsize=(15,10));a,b,c,d=axes.flat
    vals=[s['controls']['inference_ms'][f]['mean'] for f in families];errs=[s['controls']['inference_ms'][f]['stddev'] for f in families]
    bars=a.bar(range(5),vals,color=colors,yerr=errs,capsize=3)
    for bar,v,e in zip(bars,vals,errs):a.text(bar.get_x()+bar.get_width()/2,v+e+12,'%.1f'%v,ha='center',fontsize=9)
    a.set_xticks(range(5),labels,fontsize=8);a.set_ylabel('Full inference, ms (mean ± SD)');a.set_ylim(0,max(v+e for v,e in zip(vals,errs))*1.13)
    a.set_title('Detailed timers disabled; accuracy differs between models')
    groups=[('Workspace + packing + scale',('workspace','pack_pad','scale_bound','tile_plan'),'#b6cddd'),
            ('Input staging + DMA',('upload',),'#d7b25f'),('FPGA execution + polling',('completion_wait',),'#287f79'),
            ('Output DMA + copy',('download',),'#e4d4aa'),('INT8 rescale + bias',('int8_requantize','int8_accumulate'),'#c47b56'),
            ('INT8 tensor layout',('layout_allocate','input_layout','output_layout'),'#a48ec2')]
    for ax,base,title in ((b,'full',broad_label),(c,'hybrid','Only fc1 W8A8')):
        cpu=s['selected_operation_ms'][base+'_cpu']['mean'];f=base+'_fpga';total=s['selected_operation_ms'][f]['mean'];left=0
        conversions=0
        if base=='full' and meta.get('operator_details'):
            cpu=s['operator_details']['full_cpu']['complete_fc1_section_ms']['mean']
            total=s['operator_details'][f]['complete_fc1_section_ms']['mean']
            conversions=sum(s['operator_details'][f][k]['mean'] for k in ('fc1_explicit_quantize_ms','fc1_explicit_dequantize_ms'))
        ax.barh(1,cpu,color='#555e69',height=.5);ax.text(cpu+.3,1,'%.2f ms'%cpu,va='center')
        for label,keys,color in groups:
            value=sum(s['stages_ms'][f][k]['mean'] for k in keys)
            ax.barh(0,value,left=left,color=color,height=.5,label=label);left+=value
        if conversions:
            ax.barh(0,conversions,left=left,color='#d9a5bd',height=.5,label='Explicit model conversion boundaries');left+=conversions
        ax.barh(0,total-left,left=left,color='#999999',height=.5,label='Device setup, registers, MNN casts, residual')
        ax.text(total+.3,0,'%.2f ms'%total,va='center');ax.set_xlim(0,total*1.23)
        scope_labels=['CPU fc1 + model casts','FPGA fc1 + model casts'] if base=='full' and meta.get('operator_details') else ['Native INT8 CPU fc1','FPGA fc1 operator']
        ax.set_yticks([1,0],scope_labels);ax.set_xlabel('FC1 wall time, ms');ax.set_title(title)
    for base,color in (('full','#397da6'),('hybrid','#287f79')):
        changes=[r['runs'][base+'_fpga_control']['inference_ms']-r['runs'][base+'_cpu_control']['inference_ms'] for r in rows]
        d.plot(range(1,21),changes,'o-',color=color,label=broad_label if base=='full' else 'fc1-only W8A8',markersize=3)
    d.axhline(0,color='#555e69',linewidth=.8);d.set_xticks([1,5,10,15,20]);d.set_xlabel('Same random input index')
    d.set_ylabel('FPGA minus same-model CPU inference, ms');d.set_title('Isolated effect of fc1 offload');d.legend(frameon=False)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x' if ax in (b,c) else 'y',alpha=.15);ax.set_axisbelow(True)
    handles,names=b.get_legend_handles_labels();fig.legend(handles,names,loc='lower center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('TinyViT '+('corrected mixed W8A8' if repaired else 'W8A8')+': native INT8 fc1 offload, 90 MHz, 20 random images',fontsize=15)
    fig.subplots_adjust(left=.14,right=.985,top=.9,bottom=.15,wspace=.38,hspace=.6)
    for ext in ('png','svg'):fig.savefig(root/('w8a8-timing.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
