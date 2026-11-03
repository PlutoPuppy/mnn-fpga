#!/usr/bin/env python3
"""Apply the explicit DE1-SoC profile to an isolated upstream Qsys checkout."""
import argparse
import json
import pathlib
import shutil
import xml.etree.ElementTree as ET

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("qsys",type=pathlib.Path)
p.add_argument("--profile",type=pathlib.Path,default=pathlib.Path(__file__).with_name("de1-soc-matmul-profile.json"))
args=p.parse_args()
profile=json.loads(args.profile.read_text())
tree=ET.parse(args.qsys)
changes=[]
for module_name,parameters in profile["modules"].items():
    module=tree.find("./module[@name='"+module_name+"']")
    if module is None:
        raise SystemExit("Module missing: "+module_name)
    for name,value in parameters.items():
        element=module.find("./parameter[@name='"+name+"']")
        if element is None:
            raise SystemExit("Parameter missing: "+module_name+"."+name)
        old=element.get("value")
        if old!=value:
            changes.append((module_name+"."+name,old,value))
            element.set("value",value)
if changes:
    backup=args.qsys.with_suffix(".qsys.before-mnn-profile")
    if not backup.exists():
        shutil.copy2(args.qsys,backup)
    tree.write(args.qsys,encoding="utf-8",xml_declaration=True)
for name,old,new in changes:
    print(name+": "+str(old)+" -> "+new)
print("Profile applied; regenerate Qsys and rebuild before using the image.")
qsf=args.qsys.with_suffix(".qsf")
text=qsf.read_text()
old="set_global_assignment -name SDC_FILE adder_prototype.out.sdc"
new="set_global_assignment -name SDC_FILE mnn_matmul_soc.sdc"
if old in text:
    backup=qsf.with_suffix(".qsf.before-mnn-profile")
    if not backup.exists():
        shutil.copy2(qsf,backup)
    qsf.write_text(text.replace(old,new))
elif new not in text:
    raise SystemExit("Expected upstream SDC assignment missing: "+str(qsf))
shutil.copy2(pathlib.Path(__file__).with_name("matmul_soc.sdc"),args.qsys.parent/"mnn_matmul_soc.sdc")
print("Timing constraints use derived PLL clocks instead of the upstream exported snapshot.")
