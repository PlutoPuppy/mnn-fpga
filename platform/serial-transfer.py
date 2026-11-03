#!/usr/bin/env python3
"""Transfer one file to an already logged-in Linux serial shell, with SHA256.

Requires pyserial on the host and base64/sha256sum on the board. Close PuTTY
first. Uses text-safe chunks; no reboot, network setup, or FPGA access.
"""
import argparse
import base64
import hashlib
import pathlib
import shlex
import time
import uuid
import serial


def wait_line(port, wanted, seconds):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        line=port.readline().decode("utf-8",errors="replace").strip()
        if line==wanted:
            return
    raise TimeoutError("No serial shell acknowledgement: "+wanted)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=pathlib.Path)
    p.add_argument("destination",help="Absolute Linux destination path")
    p.add_argument("--port",default="COM8")
    p.add_argument("--baud",type=int,default=115200)
    args=p.parse_args()
    if not args.destination.startswith("/"):
        p.error("destination must be absolute")
    data=args.source.read_bytes()
    digest=hashlib.sha256(data).hexdigest()
    token="CODEX_SERIAL_"+uuid.uuid4().hex
    part=args.destination+"."+token+".part"
    encoded=base64.b64encode(data)
    with serial.Serial(args.port,args.baud,timeout=0.5,write_timeout=10) as port:
        # The ready marker is a separate line, so the initial terminal echo
        # cannot be mistaken for acknowledgement that the shell is ready.
        port.write(("stty -echo; printf '\\n"+token+"\\n'\n").encode())
        try:
            wait_line(port,token,10)
            port.write(("base64 -d > "+shlex.quote(part)+" <<'"+token+"'\n").encode())
            for start in range(0,len(encoded),1024):
                port.write(encoded[start:start+1024]+b"\n")
                port.flush()
                time.sleep(0.015)
                if start%(64*1024)==0:
                    print(f"Transferred {min(start+1024,len(encoded))}/{len(encoded)} encoded bytes",flush=True)
            port.write((token+"\nsha256sum "+shlex.quote(part)+"\n").encode())
            deadline=time.monotonic()+30
            found=False
            while time.monotonic()<deadline:
                line=port.readline().decode("utf-8",errors="replace").strip()
                # An interactive shell can prefix the result with its prompt,
                # even with input echo disabled. The digest isn't in our command.
                if digest+" " in line:
                    found=True
                    break
            if not found:
                raise RuntimeError("SHA256 not confirmed; destination not replaced, partial file retained: "+part)
            port.write(("mv -- "+shlex.quote(part)+" "+shlex.quote(args.destination)+
                        " && printf '\\n"+token+"_DONE\\n'\n").encode())
            wait_line(port,token+"_DONE",10)
            print(f"Verified {len(data)} bytes SHA256={digest}",flush=True)
        finally:
            port.write(b"\x03stty echo\n")
            port.flush()


if __name__=="__main__":
    main()
