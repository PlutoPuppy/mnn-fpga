#!/usr/bin/env python3
"""Check captured native MNN fc1 tensors against independent integer oracles."""
import argparse,json,pathlib
import numpy as np

def metrics(reference,actual):
    a=np.asarray(reference,dtype=np.float64);b=np.asarray(actual,dtype=np.float64);d=b-a
    return dict(max_abs=float(np.abs(d).max()),rmse=float(np.sqrt(np.mean(d*d))),
                relative_l2=float(np.linalg.norm(d)/np.linalg.norm(a)),mean_signed_error=float(d.mean()),
                exact_values=int(np.count_nonzero(a==b)),values=int(a.size))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=pathlib.Path);root=p.parse_args().directory
    result={}
    for family in ('full','hybrid'):
        cpu=root/(family+'-cpu');fpga=root/(family+'-fpga');m,k,n=784,128,512
        a=np.fromfile(str(cpu)+'.input.bin',dtype=np.int8).reshape(m,k).astype(np.int32)
        a_hw=np.fromfile(str(fpga)+'.input.bin',dtype=np.int8).reshape(m,k).astype(np.int32)
        assert np.array_equal(a,a_hw),'CPU/FPGA native INT8 activations differ'
        b=np.fromfile(str(fpga)+'.weights.bin',dtype=np.int8).reshape(k,n).astype(np.int32)
        gains=np.fromfile(str(fpga)+'.multipliers.f64',dtype='<f8');bias=np.fromfile(str(fpga)+'.biasq16.i64',dtype='<i8')
        native=np.fromfile(str(cpu)+'.output.bin',dtype=np.int8).reshape(m,n)
        actual=np.fromfile(str(fpga)+'.output.bin',dtype=np.int8).reshape(m,n)
        raw=a@b
        real=raw*gains[None,:]+bias[None,:]/65536
        full=np.clip(np.sign(real)*np.floor(np.abs(real)+.5),-127,127).astype(np.int8)
        stats=metrics(native,full)
        assert stats['max_abs']<=1,'Native MNN fc1 differs from independent full INT32 accumulation'
        expected=np.zeros((m,n),dtype=np.int8);scales=[]
        for r in range(0,m,128):
            for col in range(0,n,128):
                qa=a[r:r+128];qb=b[:,col:col+128];bands=k//16
                bound=int(np.abs(qa).sum(axis=1).max())*int(np.abs(qb).max())
                q=min(65535,(127-bands)*65536//bound) if bound else 65535;scales.append(q)
                total=np.zeros((qa.shape[0],qb.shape[1]),dtype=np.int64)
                for band in range(0,k,16):
                    product=qa[:,band:band+16].astype(np.int64)@qb[band:band+16].astype(np.int64)
                    partial=np.clip((product*q)//65536,-128,127)
                    total=np.clip(total+partial,-128,127)
                gainq=np.floor(gains[col:col+128]*4294967296.0/q+.5).astype(np.int64)
                units=total*gainq[None,:]+bias[None,col:col+128]
                rounded=np.sign(units)*((np.abs(units)+32768)//65536)
                expected[r:r+qa.shape[0],col:col+qb.shape[1]]=np.clip(rounded,-127,127).astype(np.int8)
        assert np.array_equal(expected,actual),'FPGA output differs from independent banded + Q16 requantization oracle'
        result[family]=dict(native_int8_inputs_identical=True,hardware_matches_banded_oracle=True,
            independent_full_int32_vs_mnn=stats,hardware_vs_mnn=metrics(native,actual),hardware_vs_full_int32=metrics(full,actual),
            hardware_q16_scales=scales,per_channel_multipliers=dict(minimum=float(gains.min()),maximum=float(gains.max())))
        print('PASS',family,'native INT32 oracle maxerr',stats['max_abs'],'hardware matches banded oracle exactly')
    (root/'numeric-summary.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
