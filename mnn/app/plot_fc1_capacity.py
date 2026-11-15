#!/usr/bin/env python3
"""Export measured CPU/fixed/capacity fc1 timing figures."""
import argparse,json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=pathlib.Path)
    root=p.parse_args().directory;s=json.loads((root/'summary.json').read_text())
    rows=json.loads((root/'results.json').read_text());families=('cpu','fixed','capacity')
    labels=('HPS CPU','Fixed K=64','Capacity K=128');colors=('#545e6b','#7998c5','#287f79')
    fig,(a,b,c)=plt.subplots(1,3,figsize=(17,6),gridspec_kw={'width_ratios':[1,1.5,1.15]})
    values=[s['control_inference'][f]['mean'] for f in families]
    errors=[s['control_inference'][f]['stddev'] for f in families]
    bars=a.bar(labels,values,color=colors,width=.6,yerr=errors,capsize=3)
    for bar,v in zip(bars,values):a.text(bar.get_x()+bar.get_width()/2,v+12,'%.2f'%v,ha='center')
    a.set_ylim(0,max(v+e for v,e in zip(values,errors))*1.14);a.set_ylabel('Full inference (ms), detailed timers disabled')
    a.set_title('Mean ± SD, 20 random images, 90 MHz')
    groups=[('Tensor layout + allocation',('layout_allocate','input_layout','output_layout'),'#a58fc0'),
            ('Input peak scan + INT8 quantization',('peak_scan','quantize'),'#8db5d5'),
            ('Workspace + packing + scaling',('workspace','pack_pad','scale_bound','tile_plan'),'#c3d9e6'),
            ('Input staging + DMA',('upload',),'#d8b45f'),
            ('FPGA execution + polling',('completion_wait',),'#287f79'),
            ('Output DMA + copy',('download',),'#e2d1a8'),
            ('Reconstruct + convert FP32',('dequantize','output_convert'),'#c47c57')]
    b.barh(2,s['selected_operation_ms']['cpu']['mean'],color=colors[0],height=.5,label='CPU whole fc1')
    b.text(s['selected_operation_ms']['cpu']['mean']+.8,2,'%.2f ms'%s['selected_operation_ms']['cpu']['mean'],va='center')
    for y,f in ((1,'fixed'),(0,'capacity')):
        phase=s['stages_ms'][f];left=0
        for label,keys,color in groups:
            value=sum(phase[k]['mean'] for k in keys)
            b.barh(y,value,left=left,color=color,height=.5,label=label if y==1 else None);left+=value
        total=s['selected_operation_ms'][f]['mean']
        b.barh(y,total-left,left=left,color='#999999',height=.5,label='Setup, registers, residual' if y==1 else None)
        b.text(total+.8,y,'%.2f ms'%total,va='center')
    b.set_yticks([2,1,0],labels);b.set_xlabel('Selected fc1 wall time (ms)');b.set_title('Complete selected-operation pipeline')
    b.set_xlim(0,max(s['selected_operation_ms'][f]['mean'] for f in families)*1.22)
    for f,color in zip(('fixed','capacity'),colors[1:]):
        delta=[r['runs'][f+'_control']['inference_ms']-r['runs']['cpu_control']['inference_ms'] for r in rows]
        c.plot(range(1,21),delta,'o-',color=color,label=f.capitalize(),markersize=3)
    c.axhline(0,color=colors[0],linewidth=.8);c.set_xticks([1,5,10,15,20])
    c.set_xlabel('Random input index');c.set_ylabel('Offload minus CPU inference (ms)');c.legend(frameon=False)
    c.set_title('All pairs; CPU outlier retained')
    for ax in (a,b,c):
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x' if ax==b else 'y',alpha=.15);ax.set_axisbelow(True)
    handles,names=b.get_legend_handles_labels()
    fig.legend(handles,names,loc='lower center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('TinyViT: stage-1 block-0 MLP fc1 only, 784 x 128 times 128 x 512',fontsize=14)
    fig.subplots_adjust(left=.055,right=.985,top=.82,bottom=.28,wspace=.42)
    for ext in ('png','svg'):fig.savefig(root/('fc1-timing.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
