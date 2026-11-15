#!/usr/bin/env python3
"""Summarize the board's paired benchmark, including paired bootstrap intervals."""
import argparse
import json
import pathlib
import random
import statistics


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values)-1)*fraction
    lo = int(position)
    hi = min(lo+1, len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def stats(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                stddev=statistics.stdev(values), minimum=min(values), maximum=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=pathlib.Path)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    directory = args.directory
    rows = json.loads((directory/'results.json').read_text())
    metadata = json.loads((directory/'metadata.json').read_text())
    assert len(rows) == metadata['pairs'] == 20
    assert len({r['image_sha256'] for r in rows}) == 20
    assert all(r['cpu']['exit_code'] == r['fpga']['exit_code'] == 0 for r in rows)
    assert all(r['cpu']['fpga_jobs'] == 0 and r['fpga']['fpga_jobs'] == metadata['expected_jobs_per_fpga_inference'] for r in rows)
    assert sum(r['order'][0] == 'cpu' for r in rows) == 10
    summary = {'pairs': len(rows), 'measured_inferences': 2*len(rows),
               'measured_fpga_jobs': sum(r['fpga']['fpga_jobs'] for r in rows),
               'cpu_first_pairs': 10, 'fpga_first_pairs': 10,
               'bootstrap_replicates': 20000, 'bootstrap_seed': 20261111}
    for metric in ('inference_ms', 'process_ms'):
        cpu = [r['cpu'][metric] for r in rows]
        fpga = [r['fpga'][metric] for r in rows]
        delta = [a-b for a,b in zip(cpu,fpga)]
        rng = random.Random(summary['bootstrap_seed'])
        boot_delta, boot_reduction = [], []
        for _ in range(summary['bootstrap_replicates']):
            ids = [rng.randrange(len(rows)) for _ in rows]
            a, b = sum(cpu[i] for i in ids), sum(fpga[i] for i in ids)
            boot_delta.append((a-b)/len(rows))
            boot_reduction.append(100*(a-b)/a)
        item = {'cpu': stats(cpu), 'fpga': stats(fpga), 'saved_ms': stats(delta),
                'speedup_ratio_of_means': sum(cpu)/sum(fpga),
                'time_reduction_pct_ratio_of_means': 100*(sum(cpu)-sum(fpga))/sum(cpu),
                'offload_faster_pairs': sum(v > 0 for v in delta),
                'offload_slower_pairs': sum(v < 0 for v in delta),
                'paired_bootstrap_95_ci_saved_ms': [quantile(boot_delta,.025),quantile(boot_delta,.975)],
                'paired_bootstrap_95_ci_time_reduction_pct': [quantile(boot_reduction,.025),quantile(boot_reduction,.975)]}
        for first in ('cpu','fpga'):
            subset = [r for r in rows if r['order'][0] == first]
            item[first+'_first_mean_saved_ms'] = statistics.mean(r['cpu'][metric]-r['fpga'][metric] for r in subset)
        summary[metric] = item
    summary['scores'] = {
        'top1_agreement_pairs':sum(r['scores']['top1_agrees'] for r in rows),
        'top5_ranking_agreement_pairs':sum(r['scores']['top5_ranking_agrees'] for r in rows),
        'max_abs_error': stats([r['scores']['max_abs_error'] for r in rows]),
        'cosine_similarity': stats([r['scores']['cosine_similarity'] for r in rows])}
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['| Pair | First | CPU inference ms | FPGA inference ms | Time reduction | CPU process ms | FPGA process ms |',
             '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append('| %d | %s | %.2f | %.2f | %+.2f%% | %.2f | %.2f |' %
                     (r['pair'],r['order'][0].upper(),r['cpu']['inference_ms'],r['fpga']['inference_ms'],
                      r['inference_ms']['reduction_pct'],r['cpu']['process_ms'],r['fpga']['process_ms']))
    (directory/'pairs.md').write_text('\n'.join(lines)+'\n')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,2,figsize=(11,4.4))
        x = [r['pair'] for r in rows]
        for ax,metric,title in zip(axes,('inference_ms','process_ms'),('Full-model inference','Complete process wall time')):
            ax.plot(x,[r['cpu'][metric] for r in rows],'-o',label='HPS CPU only',color='#2672a5',markersize=3)
            ax.plot(x,[r['fpga'][metric] for r in rows],'-o',label='First QK on FPGA',color='#cc5d24',markersize=3)
            ax.set_title(title)
            ax.set_xlabel('Paired random image')
            ax.set_ylabel('Time (ms)')
            ax.set_xticks([1,5,10,15,20])
            ax.grid(alpha=.25)
            ax.legend(loc='best',fontsize=9)
        fig.suptitle('TinyViT on DE1-SoC: 90 MHz FPGA, 20 paired random RGB images',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.94))
        fig.savefig(directory/'timings.png',dpi=180)
        fig.savefig(directory/'timings.svg')
        plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
