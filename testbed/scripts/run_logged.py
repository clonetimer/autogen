#!/usr/bin/env python3
# 用途：监督单个命令并记录失败或完成终态。
# 参数：--output --cwd --timeout -- 命令。
# 输出：stdout/stderr与status.json，超时终止该子进程组。
"""Execute one command with stdout/stderr and an atomic terminal status file.

Run this in the foreground locally. A supervising tool may poll status while this
process runs, but must not report completion until the terminal status is read.
Timeout terminates only the child process group started by this program.
"""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--cwd',required=True)
    p.add_argument('--timeout',type=float,default=900);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not cmd:p.error('command required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();record={'command':cmd,'cwd':str(Path(a.cwd).resolve()),'started':datetime.now(timezone.utc).isoformat(),'status':'RUNNING'}
    write(out/'status.json',record)
    with (out/'stdout.txt').open('w') as so,(out/'stderr.txt').open('w') as se:
        process=subprocess.Popen(cmd,cwd=a.cwd,stdout=so,stderr=se,start_new_session=True)
        record['pid']=process.pid;write(out/'status.json',record)
        try:
            result=process.wait(timeout=a.timeout);record.update(status='COMPLETE' if result==0 else 'FAILED',returncode=result)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            record.update(status='INTERRUPTED',returncode=process.returncode)
    record.update(elapsed_seconds=time.monotonic()-start,ended=datetime.now(timezone.utc).isoformat())
    write(out/'status.json',record)
    return 0 if record['status']=='COMPLETE' else 1

if __name__=='__main__':raise SystemExit(main())
