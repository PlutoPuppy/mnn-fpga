#!/usr/bin/env python3
"""Run the actual NI82 output RTL against the independent C++ oracle.

Example on this Windows workspace:
 python sim/run_native_cxxrtl.py --yowasp-dir build/hdl-wasm \
   --cxx C:/intelFPGA_lite/23.1std/questa_fse/gcc-7.4.0-mingw64vc16/bin/g++.exe
"""
import argparse
import pathlib
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--yowasp-dir',type=pathlib.Path)
    p.add_argument('--yosys',default='yosys')
    p.add_argument('--cxx',default='g++')
    p.add_argument('--runtime',type=pathlib.Path,help='Yosys include/backends/cxxrtl/runtime directory')
    p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('build/native-output-sim'))
    a=p.parse_args();root=pathlib.Path(__file__).resolve().parent.parent
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    command='read_verilog -sv "'+(root/'rtl/matmul_soc/output_loader_native.sv').as_posix()+'"; '
    command+='hierarchy -top output_loader_native; proc; write_cxxrtl -header "'+(output/'native_model.cc').as_posix()+'"'
    runtime=a.runtime
    if a.yowasp_dir:
        package=a.yowasp_dir.resolve()
        code='import sys;sys.path.insert(0,'+repr(str(package))+');'
        code+='from yowasp_yosys import _run_yosys_argv;sys.argv=["yowasp-yosys","-p",sys.argv[1]];_run_yosys_argv()'
        invocation=[sys.executable,'-c',code,command]
        if runtime is None:runtime=package/'yowasp_yosys/share/include/backends/cxxrtl/runtime'
    else:
        invocation=[a.yosys,'-p',command]
        if runtime is None:p.error('Native Yosys requires --runtime')
    with (output/'yosys.log').open('w') as log:
        subprocess.run(invocation,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    if not (runtime/'cxxrtl/cxxrtl.h').is_file():p.error('Cannot find CXXRTL runtime headers')
    executable=output/('native_output_test.exe' if sys.platform=='win32' else 'native_output_test')
    subprocess.run([a.cxx,'-std=c++17','-O1','-Wall','-Wextra','-Werror','-Wno-unused-parameter',
                    '-isystem',str(runtime),'-I'+str(output),str(root/'sim/native_output_cxxrtl_test.cpp'),
                    str(output/'native_model.cc'),'-o',str(executable)],check=True)
    subprocess.run([str(executable)],check=True)


if __name__=='__main__':main()
