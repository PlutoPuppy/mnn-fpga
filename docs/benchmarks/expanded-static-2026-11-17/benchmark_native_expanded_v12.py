#!/usr/bin/env python3
"""Fresh-process paired NI82/NI83 benchmark; preserves outputs and raw logs.

Run on HPS after physical native matrix tests pass. Random images are reused from
the earlier experiment, with their original SHA256 verified before each run.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import time
from benchmark_board_random_qk import MEAN, STD, sha, scores, compare, cpu_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--board-dir', default='/home/ubuntu/tinyvit')
    p.add_argument('--source-results', default='/home/ubuntu/tinyvit/benchmark-qk-profile-90mhz-20')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--runner', required=True)
    p.add_argument('--inputs', type=int, default=20)
    p.add_argument('--experiment-date', help='Host-supplied YYYY-MM-DD; board clock may be incorrect')
    p.add_argument('--bitstream-sha256', default='not-used')
    p.add_argument('--allowlist', help='Optional measured policy, adds selective and selective_pairs modes')
    p.add_argument('--modes', default='fp32,cpu,all,pairs')
    p.add_argument('--configurations', type=pathlib.Path, help='JSON mapping extra mode names to model/offload/pairs/counts')
    p.add_argument('--abi', choices=('NI82','NI83'), default='NI82')
    p.add_argument('--natural-images', help='Optional directory of natural RGB images instead of the earlier random set')
    a = p.parse_args()
    if os.geteuid() != 0 or not 1 <= a.inputs <= 20:
        p.error('Run as root; inputs must be 1..20')
    board = pathlib.Path(a.board_dir).resolve()
    source = pathlib.Path(a.source_results).resolve()
    out = pathlib.Path(os.path.abspath(a.output_dir))
    out.mkdir(parents=True, exist_ok=False)
    modes = a.modes.split(',')
    model = 'tiny-vit-mixed-w8a8-accuracy.mnn'
    configurations={m:dict(model='tiny-vit.mnn' if m=='fp32' else model,
        offload=m not in ('fp32','cpu'),pairs='pairs' in m) for m in ('fp32','cpu','all','pairs','selective','selective_pairs')}
    if a.configurations:
        extra=json.loads(a.configurations.read_text())
        for name,cfg in extra.items():
            if not re.fullmatch('[a-z][a-z0-9_]*',name) or not isinstance(cfg.get('model'),str):p.error('Invalid configuration')
            if not isinstance(cfg.get('offload'),bool) or not isinstance(cfg.get('pairs'),bool):p.error('Explicit offload/pairs flags required')
            if cfg['pairs'] and not cfg['offload']:p.error('Resident pairs require offload')
            configurations[name]=cfg
    if len(set(modes))!=len(modes) or any(m not in configurations for m in modes):
        p.error('Unknown benchmark mode')
    if any(m.startswith('selective') for m in modes) and not a.allowlist:
        p.error('Selective mode requires allowlist')
    hardware = any(configurations[m]['offload'] for m in modes)
    if hardware and not re.fullmatch('[0-9a-f]{64}',a.bitstream_sha256):
        p.error('Hardware benchmark requires the programmed SOF SHA256')
    if a.natural_images:
        image_dir=pathlib.Path(a.natural_images).resolve()
        images=sorted(p for p in image_dir.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.ppm'))[:a.inputs]
        assert len(images)==a.inputs
        original=[dict(input=i,seed=None,image_sha256=sha(image),image_path=str(image)) for i,image in enumerate(images)]
    else:
        original = json.loads((source/'results.json').read_text())[:a.inputs]
    environment = dict(os.environ)
    for k in list(environment):
        if k.startswith('MNN_FPGA') or k.startswith('MNN_MATMUL_SOC') or k == 'MNN_INT8_TRACE':
            environment.pop(k)
    metadata = dict(date=a.experiment_date, board_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    order_design='Williams rows, including reversed rows for odd configuration counts',
                    inputs=a.inputs, modes=modes, threads=2,
                    runner=a.runner, runner_sha256=sha(board/a.runner),
                    bitstream_sha256=a.bitstream_sha256, clock_mhz=90 if hardware else None, abi=a.abi if hardware else 'not-used',
                    configurations={m:configurations[m] for m in modes},
                    models={m: sha(board/m) for m in set(configurations[x]['model'] for x in modes)},
                    profile_runs_separate=True, fresh_process=True, warmup=0,
                    source=str(source), cpu_state_before=cpu_state(),
                    timing='Inference excludes load/session/cache/LUT setup; process wall and setup also recorded')
    metadata['input_kind']='natural images' if a.natural_images else 'uniform random RGB images'
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2))

    def execute(mode, image, label, profiled):
        target = out/label/(mode+('_profile' if profiled else '_control'))
        target.mkdir(parents=True)
        cfg=configurations[mode]
        command = [str(board/a.runner), '--model', str(board/cfg['model']),
                   '--height', '224', '--width', '224', '--image', str(image), '--mean', MEAN, '--std', STD,
                   '--threads', '2', '--runs', '1', '--output-dir', str(target)]
        env = dict(environment)
        if cfg['offload']:
            env.update(MNN_MATMUL_SOC_DEVMEM='1', MNN_MATMUL_SOC_DMA='1')
            command += ['--fpga-native-int8', '--require-fpga']
            if cfg['pairs']:
                command += ['--fpga-native-pairs']
            if mode.startswith('selective'):
                command += ['--native-allowlist', str(pathlib.Path(a.allowlist).resolve())]
        if profiled:
            command += ['--profile', '--profile-ops', str(target/'operators.json')]
        with (target/'run.log').open('w') as log:
            start = time.perf_counter()
            result = subprocess.run(command, env=env, cwd=str(board), stdout=log, stderr=subprocess.STDOUT)
            wall_ms = (time.perf_counter()-start)*1000
        text = (target/'run.log').read_text()
        if result.returncode:
            raise RuntimeError('Failed '+str(target)+': '+text[-1200:])
        timing = re.findall(r'^Inference \d+: ([\d.e+-]+) ms$', text, re.M)
        jobs = re.findall(r'^FPGA submissions: (\d+)$', text, re.M)
        assert len(timing) == len(jobs) == 1
        values = scores(target/'output_0.f32')
        assert len(values) == 1000 and all(__import__('math').isfinite(x) for x in values)
        record = dict(inference_ms=float(timing[0]), process_wall_ms=wall_ms, jobs=int(jobs[0]),
                      output_sha256=sha(target/'output_0.f32'), command=command)
        if cfg['offload']:
            native = re.findall(r'^NATIVE_JSON (.+)$', text, re.M)
            assert len(native) == 1 and record['jobs'] > 0
            record['native'] = json.loads(native[0])
            assert record['native']['pending'] == -1
            assert record['native']['abi']==a.abi
            if 'operations' in cfg:assert record['native']['operations']==cfg['operations']
            if 'pair_count' in cfg:assert record['native']['fused_pairs']==cfg['pair_count']
        else:
            assert record['jobs'] == 0
        if profiled:
            record['profile'] = json.loads(re.findall(r'^PROFILE_JSON (.+)$', text, re.M)[0])
            if 'native' in record:
                assert record['profile']['offload']['local_bytes'] == record['native']['local_feedback_bytes']
        return record, values

    rows = []
    for position,entry in enumerate(original):
        i = entry['input']
        image = pathlib.Path(entry['image_path']) if 'image_path' in entry else source/'images'/('random-%02d.ppm' % i)
        assert sha(image) == entry['image_sha256']
        base = [0]
        for v in range(1,len(modes)):
            base.append((v+1)//2 if v%2 else len(modes)-v//2)
        # Odd-sized Williams designs also need the reversed rows to balance
        # preceding-model/cache effects. Even-sized designs need only rotation.
        if len(modes)%2 and (position//len(modes))%2:
            base=list(reversed(base))
        order = [modes[(v+position)%len(modes)] for v in base]
        runs, outputs = {}, {}
        for profiled in (False, True):
            for mode in order if not profiled else list(reversed(order)):
                key = mode+('_profile' if profiled else '_control')
                runs[key], outputs[key] = execute(mode, image, 'input-%02d' % i, profiled)
        for mode in modes:
            assert runs[mode+'_control']['output_sha256'] == runs[mode+'_profile']['output_sha256']
        checks = {}
        if 'cpu' in modes:
            for mode in modes:
                if mode != 'cpu':
                    checks[mode+'_vs_cpu'] = compare(outputs['cpu_control'], outputs[mode+'_control'])
        if 'all' in modes and 'pairs' in modes:
            checks['pairs_vs_all'] = compare(outputs['all_control'], outputs['pairs_control'])
            assert runs['all_control']['output_sha256'] == runs['pairs_control']['output_sha256'], 'Resident fusion changed output bytes'
        for mode in modes:
            cfg=configurations[mode]
            for reference in modes:
                other=configurations[reference]
                if mode==reference or cfg['model']!=other['model']:continue
                if cfg['offload'] and not other['offload']:
                    checks[mode+'_vs_'+reference]=compare(outputs[reference+'_control'],outputs[mode+'_control'])
                if cfg['offload'] and cfg['pairs'] and other['offload'] and not other['pairs']:
                    assert runs[mode+'_control']['output_sha256']==runs[reference+'_control']['output_sha256'],'Resident fusion changed output bytes'
        rows.append(dict(input=i, seed=entry['seed'], image_sha256=entry['image_sha256'], image_path=str(image),
                         order=order, runs=runs, scores=checks))
        (out/'results.json').write_text(json.dumps(rows, indent=2))
        print('INPUT %02d ' % i+'; '.join('%s %.3f ms' % (m, runs[m+'_control']['inference_ms']) for m in modes), flush=True)
    metadata['cpu_state_after'] = cpu_state()
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
    print('COMPLETE',len(rows),'images;',len(rows)*len(modes)*2,'measured inferences',flush=True)


if __name__ == '__main__':
    main()
