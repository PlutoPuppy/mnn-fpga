#!/usr/bin/env python3
"""Download one Linux file over an already logged-in serial shell, with SHA256."""
import argparse
import base64
import hashlib
import pathlib
import shlex
import time
import uuid
import serial


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",help="Absolute Linux source path")
    p.add_argument("destination",type=pathlib.Path)
    p.add_argument("--port",default="COM8")
    args=p.parse_args()
    if not args.source.startswith("/") or args.destination.exists():
        p.error("use an absolute remote source and a new local destination")
    token="CODEX_DOWNLOAD_"+uuid.uuid4().hex
    command="printf '\\n"+token+"\\n'; base64 "+shlex.quote(args.source)+"; printf '\\n"+token+"_END\\n'; sha256sum "+shlex.quote(args.source)+"\n"
    data=[]
    started=False
    ended=False
    with serial.Serial(args.port,115200,timeout=0.5,write_timeout=10) as port:
        port.write(command.encode())
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            line=port.readline().decode("ascii",errors="replace").strip()
            if line==token:
                started=True
            elif line==token+"_END":
                ended=True
            elif started and not ended and line:
                data.append(line)
            elif ended and line:
                decoded=base64.b64decode("".join(data),validate=True)
                if line.startswith(hashlib.sha256(decoded).hexdigest()+" "):
                    args.destination.parent.mkdir(parents=True,exist_ok=True)
                    args.destination.write_bytes(decoded)
                    print(f"Verified {len(decoded)} bytes: {args.destination}")
                    return
                raise RuntimeError("Remote SHA256 mismatch or file read error; local destination not written")
        raise TimeoutError("Serial download incomplete; local destination not written")


if __name__=="__main__":
    main()
