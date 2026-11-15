#!/usr/bin/env python3
"""Standalone figures for whole-model CPU operator measurements."""
import argparse,json,pathlib,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def label(name):
    name=re.sub(r'/Add_output_0__matmul_converted$','',name)
    name=re.sub(r'/(Conv|Mul_1)_output_0$','',name)
    name=name.replace('/layers.','L').replace('/blocks.','/B')
    return name


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=pathlib.Path);a=p.parse_args();root=a.directory
    summary=json.loads((root/'summary.json').read_text());ranking=json.loads((root/'operator-ranking.json').read_text())
    items=summary['types'];fig,ax=plt.subplots(figsize=(9,5.5))
    bars=ax.barh([x['type'] for x in items],[x['ms']['mean'] for x in items],color='#3974b9')
    ax.invert_yaxis();ax.set_xlim(0,max(x['ms']['mean'] for x in items)*1.23)
    for b,x in zip(bars,items):ax.text(b.get_width()+5,b.get_y()+b.get_height()/2,'%.1f ms / %.1f%%'%(x['ms']['mean'],x['share_pct']),va='center',fontsize=9)
    ax.set_xlabel('Mean operator wall time per inference (ms), 20 random images')
    ax.set_title('TinyViT CPU time by runtime operator type (2 HPS threads)')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(root/('operator-types.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    items=ranking[:20];fig,ax=plt.subplots(figsize=(13,8.5))
    colors=['#3974b9' if x['kind']=='static_pointwise_gemm' else '#dd9850' if x['kind']=='spatial_convolution' else '#8c839b' for x in items]
    bars=ax.barh([label(x['name']) for x in items],[x['ms']['mean'] for x in items],color=colors,
                 xerr=[x['ms']['stddev'] for x in items],error_kw=dict(elinewidth=.8,capsize=2))
    ax.invert_yaxis();ax.set_xlim(0,max(x['ms']['mean']+x['ms']['stddev'] for x in items)*1.22)
    for b,x in zip(bars,items):ax.text(b.get_width()+x['ms']['stddev']+.4,b.get_y()+b.get_height()/2,'%.2f ms'%x['ms']['mean'],va='center',fontsize=9)
    ax.set_xlabel('Mean runtime operator wall time (ms); error bars = sample SD across 20 images')
    ax.set_title('Largest CPU operators: blue = static 1x1 GEMM, orange = spatial convolution')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(root/('top-operators.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
