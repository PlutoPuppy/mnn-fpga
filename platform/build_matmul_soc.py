#!/usr/bin/env python3
"""Generate/build the upstream Quartus project; never programs the board.

--native-hps-launcher creates a temporary installation view using directory
junctions and a local Nios command launcher. The installed files are unchanged.
It runs Intel's native HPS ROM compiler, not general Nios/WSL compilation.
"""
import argparse
import pathlib
import shutil
import os
import subprocess
import tempfile
import sys
from check_quartus_timing import require_timing


def installation_view(source,destination,exceptions):
    destination.mkdir(parents=True,exist_ok=True)
    for item in source.iterdir():
        target=destination/item.name
        if item.name in exceptions:
            if item.is_dir():
                installation_view(item,target,exceptions[item.name])
            continue
        if target.exists():
            continue
        if item.is_dir():
            subprocess.run(["cmd","/c","mklink","/J",str(target),str(item)],check=True,capture_output=True)
        else:
            shutil.copy2(item,target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quartus-root",type=pathlib.Path,default=pathlib.Path("C:/intelFPGA_lite/23.1std/quartus"))
    p.add_argument("--friend-root",type=pathlib.Path,required=True)
    p.add_argument("--native-hps-launcher",action="store_true")
    p.add_argument("--generate-only",action="store_true")
    p.add_argument("--skip-generation",action="store_true",help="Use previously generated synthesis files")
    args=p.parse_args()
    repo=pathlib.Path(__file__).resolve().parent.parent
    build=repo/"build"
    build.mkdir(exist_ok=True)
    friend=args.friend_root.resolve()
    quartus=args.quartus_root.resolve()
    if not (friend/"adder_prototype.qpf").is_file():
        p.error("friend-root must contain adder_prototype.qpf")
    native=(quartus/"../nios2eds/bin/gnu/H-x86_64-mingw32/bin").resolve()
    env=dict(os.environ)
    if args.native_hps_launcher and not args.skip_generation:
        mirror=pathlib.Path(tempfile.gettempdir())/"mnn-fpga-quartus-native"
        exceptions={"quartus":{"sopc_builder":{"bin":{}}},
                    "nios2eds":{"Nios II Command Shell.bat":{}}}
        installation_view(quartus.parent,mirror,exceptions)
        launcher=mirror/"nios2eds/Nios II Command Shell.bat"
        helper=repo/"platform/native_hps_make.py"
        launcher.write_text('@echo off\nset "PATH='+str(native)+';%PATH%"\n"'+sys.executable+'" "'+str(helper)+'" %*\n')
        quartus=mirror/"quartus"
        env["QUARTUS_ROOTDIR"]=str(quartus)
    if not args.skip_generation:
        subprocess.run([str(quartus/"sopc_builder/bin/qsys-generate.exe"),"adder_prototype.qsys","--synthesis=VERILOG"],cwd=friend,env=env,check=True)
    if not args.generate_only:
        steps=[("quartus_map.exe",["adder_prototype"]),
               ("quartus_sta.exe",["-t","adder_prototype/synthesis/submodules/hps_sdram_p0_pin_assignments.tcl","adder_prototype"]),
               ("quartus_fit.exe",["adder_prototype"]),
               ("quartus_asm.exe",["adder_prototype"]),
               ("quartus_sta.exe",["adder_prototype"])]
        for exe,options in steps:
            log=friend/(exe+".log")
            print("Running "+exe+"; log: "+str(log),flush=True)
            with log.open("w") as output:
                result=subprocess.run([str(quartus/"bin64"/exe),*options],cwd=friend,stdout=output,stderr=subprocess.STDOUT)
            if result.returncode:
                print("\n".join(log.read_text(errors="replace").splitlines()[-25:]),flush=True)
                raise SystemExit(result.returncode)
        require_timing(friend/"output_files/adder_prototype.sta.summary")


if __name__=="__main__":
    main()
