#!/usr/bin/env python3
# 用途：单包限额校验与原子暂存；参数：probe/stage与路径/哈希；输出：JSON探针或暂存回执。
"""Post-entry staging: one allowlisted archive, bounded checks, atomic commit.

This cannot change ChatGPT's pre-execution attachment hydration or restart CAAS.
A successful first command proves entry availability only, not prior root cause.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def probe(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    value = {'schema':'phm.caas-probe.v1', 'python':platform.python_version(),
             'architecture':platform.machine(), 'calculation':sum(range(10)),
             'pid':os.getpid(), 'time_unix':time.time(),
             'pre_execution_hydration_controlled':False,
             'historical_ClientError_root_cause_confirmed':False}
    fd, name = tempfile.mkstemp(prefix='probe_', suffix='.json', dir=directory)
    with os.fdopen(fd, 'w') as f:
        json.dump(value, f); f.flush(); os.fsync(f.fileno())
    path=Path(name)
    value['file_roundtrip'] = json.loads(path.read_text()) == value
    value['probe_file']=str(path)
    return value


def stage(archive:Path,destination:Path,expected_sha256:str,*,max_bytes=64*1024*1024,
          max_entries=10000,max_expanded_bytes=256*1024*1024,max_ratio=300) -> dict:
    """No recursive discovery, overwrite, nested extraction, network or imports."""
    start=time.monotonic()
    for value in (max_bytes,max_entries,max_expanded_bytes,max_ratio):
        if type(value) is not int or value<=0: raise ValueError('INVALID_BUDGET')
    if destination.exists(): raise FileExistsError('STAGING_DESTINATION_EXISTS')
    if not archive.is_file() or archive.stat().st_size>max_bytes:
        raise ValueError('ARCHIVE_MISSING_OR_COMPRESSED_BUDGET_EXCEEDED')
    digest=sha(archive)
    if digest!=expected_sha256: raise ValueError('INPUT_SHA256_MISMATCH')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix=destination.name+'.partial-',dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as z:
            members=z.infolist()
            if not members or len(members)>max_entries: raise ValueError('ENTRY_BUDGET')
            names=set(); expanded=0
            for m in members:
                p=PurePosixPath(m.filename)
                if ('\\' in m.filename or p.is_absolute() or '..' in p.parts or ':' in m.filename
                    or not p.parts or p.as_posix() in names): raise ValueError('UNSAFE_OR_DUPLICATE_PATH')
                names.add(p.as_posix())
                mode=m.external_attr>>16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0,stat.S_IFREG,stat.S_IFDIR)):
                    raise ValueError('SPECIAL_FILE_FORBIDDEN')
                if m.flag_bits&1: raise ValueError('ENCRYPTED_ARCHIVE_UNSUPPORTED')
                expanded+=m.file_size
                if expanded>max_expanded_bytes or m.file_size>max(1,m.compress_size)*max_ratio:
                    raise ValueError('EXPANSION_BUDGET')
            written=0
            for m in members:
                target=temp.joinpath(*PurePosixPath(m.filename).parts)
                if m.is_dir(): target.mkdir(parents=True,exist_ok=True); continue
                target.parent.mkdir(parents=True,exist_ok=True)
                count=0
                with z.open(m) as source,target.open('xb') as output:
                    for block in iter(lambda:source.read(1024*1024),b''):
                        count+=len(block); written+=len(block)
                        if count>m.file_size or written>max_expanded_bytes:
                            raise ValueError('STREAM_EXPANSION_BUDGET')
                        output.write(block)
                if count!=m.file_size:raise ValueError('STREAM_SIZE_MISMATCH')
            os.rename(temp,destination)
        return {'schema':'phm.stage-receipt.v1','archive_name':archive.name,
                'sha256':digest,'compressed_bytes':archive.stat().st_size,
                'expanded_bytes':expanded,'entries':len(members),
                'destination':str(destination),'status':'STAGED_AND_CRC_VERIFIED',
                'elapsed_seconds':time.monotonic()-start,
                'nested_archives_extracted':False,'global_files_deleted':False}
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__); s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('probe'); a.add_argument('--directory',type=Path,required=True)
    a=s.add_parser('stage'); a.add_argument('--archive',type=Path,required=True)
    a.add_argument('--destination',type=Path,required=True); a.add_argument('--sha256',required=True)
    a.add_argument('--max-mib',type=int,default=64)
    args=p.parse_args()
    result=(probe(args.directory) if args.command=='probe' else stage(args.archive,args.destination,args.sha256,max_bytes=args.max_mib*1024*1024))
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
