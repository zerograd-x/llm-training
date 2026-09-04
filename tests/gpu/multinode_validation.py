"""Run paired node drivers against a new shared output directory."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

CASES=('runtime_ddp','lora_zero3','lora_zero3_gc','tt_zero3','tt_zero3_gc')

def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def wait_for(path,timeout=1000):
    end=time.monotonic()+timeout
    while not path.exists():
        if time.monotonic()>end:raise TimeoutError(str(path))
        time.sleep(1)
    return json.loads(path.read_text())

def run_command(cmd, env, log, timeout=300):
    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        return 124

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--node-rank',type=int,choices=[0,1],required=True);p.add_argument('--master',required=True);p.add_argument('--cases',nargs='+',choices=CASES,default=CASES)
    args=p.parse_args();root=args.root;node=args.node_rank
    root.mkdir(parents=True,exist_ok=True)
    marker=root/f'start-node-{node}.json'
    if marker.exists():raise RuntimeError('Use a new output directory')
    atomic(marker,{'node':node})
    worker=Path(__file__).with_name('multirank_case.py');env=dict(os.environ)
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT','ACCELERATE_USE_CPU','ACCELERATE_USE_DEEPSPEED'):env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',NCCL_IB_DISABLE='1',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    results=[]
    for case in args.cases:
        folder=root/case;folder.mkdir(exist_ok=True)
        for phase in ('prepare','baseline','interrupt','resume','verify'):
            started=time.monotonic()
            if phase in ('prepare','verify'):
                if node==0:
                    with (folder/f'{phase}.log').open('w') as log:
                        code=run_command([sys.executable,str(worker),case,phase,str(root)],env,log)
                    payload=json.loads((folder/f'{phase}.json').read_text()) if (folder/f'{phase}.json').exists() else {}
                    atomic(folder/f'{phase}-done.json',{'ok':code==0 and payload.get('ok') is True,'returncode':code})
                status=wait_for(folder/f'{phase}-done.json')
            else:
                phase_env=dict(env)
                if case!='runtime_ddp':phase_env['ACCELERATE_USE_DEEPSPEED']='true'
                cmd=[sys.executable,'-m','torch.distributed.run','--nnodes=2','--nproc_per_node=1',f'--node_rank={node}',f'--master_addr={args.master}','--master_port=29501',str(worker),case,phase,str(root)]
                with (folder/f'{phase}-node-{node}.log').open('w') as log:
                    code=run_command(cmd,phase_env,log)
                payload_path=folder/f'{phase}-rank-{node}.json'
                payload=json.loads(payload_path.read_text()) if payload_path.exists() else {}
                atomic(folder/f'{phase}-exit-node-{node}.json',{'ok':code==0 and payload.get('ok') is True,'returncode':code})
                statuses=[wait_for(folder/f'{phase}-exit-node-{i}.json') for i in range(2)]
                status={'ok':all(x['ok'] for x in statuses),'nodes':statuses}
            result=dict(case=case,phase=phase,**status,elapsed_seconds=round(time.monotonic()-started,3));results.append(result)
            if node==0:atomic(root/'matrix.json',results);print(json.dumps(result),flush=True)
            if not result['ok']:break
    return 0 if all(x['ok'] for x in results) else 1

if __name__=='__main__':sys.exit(main())

