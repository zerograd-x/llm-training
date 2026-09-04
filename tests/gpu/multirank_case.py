"""Two-rank CUDA/NCCL correctness and artifact lifecycle with shared storage."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import traceback
from unittest.mock import patch
import warnings

import torch
from transformers import AutoModelForCausalLM
from llm_training import (Trainer, DistributedRuntime, TrainingConfig, DeepSpeedConfig,
    CheckpointConfig, OptimizerConfig, SchedulerConfig, LoRAConfig,
    TrainableTokensConfig, GradientCheckpointingConfig, require_single_process)
import fresh_case as single
from tt_probe import effective_rows, assert_token_rows


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class Rows(torch.utils.data.Dataset):
    def __init__(self, tokens):
        self.rows = [[1, 3+i, 32+i%2 if tokens else 12+i, 2] for i in range(8)]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        x = torch.tensor(self.rows[i])
        return dict(input_ids=x, attention_mask=torch.ones_like(x), labels=x.clone())


class Stop(RuntimeError): pass


class Probe(Trainer):
    stop = False
    root = None
    def save_checkpoint(self):
        path = super().save_checkpoint()
        if self.stop and self.state.global_step == 1:
            rank = self.runtime.process_index
            single.save(self.root / f"checkpoint-rng-{rank}.pt", single.rng())
            single.save(self.root / f"checkpoint-params-{rank}.pt", single.snapshot(self))
            single.save(self.root / f"checkpoint-scheduler-{rank}.pt", self.scheduler.state_dict())
            raise Stop()
        return path


def config(case, root, resume):
    return TrainingConfig(output_dir=str(root), num_epochs=1, seed=123,
        mixed_precision="bf16", per_device_batch_size=1, gradient_accumulation_steps=2,
        log_every_steps=1,
        optimizer=OptimizerConfig(learning_rate=0.01, weight_decay=0),
        scheduler=SchedulerConfig(name="linear", warmup_steps=0),
        lora=LoRAConfig(rank=4, alpha=8, target_modules=("c_attn",)) if case.startswith("lora") else None,
        trainable_tokens=TrainableTokensConfig(n_tokens_added=2) if case.startswith("tt") else None,
        gradient_checkpointing=GradientCheckpointingConfig(enabled=case.endswith("gc"),use_reentrant=False),
        deepspeed=DeepSpeedConfig(zero_stage=3) if case != "runtime_ddp" else None,
        checkpoint=CheckpointConfig(save_every_steps=1, keep_last=2,resume_from="latest" if resume else None))


def train(case, phase, root):
    rank = int(os.environ["RANK"])
    cfg = config(case, root/("baseline-run" if phase=="baseline" else "resumed-run"), phase=="resume")
    runtime = DistributedRuntime(cfg)
    assert runtime.num_processes == 2 and runtime.process_index == rank
    assert runtime.local_process_index == int(os.environ["LOCAL_RANK"])
    assert runtime.is_main_process == (rank == 0)
    assert runtime.is_local_main_process == (runtime.local_process_index == 0)
    assert torch.distributed.get_backend() == "nccl"
    assert runtime.accelerator.device.type == "cuda" and torch.cuda.is_bf16_supported()
    assert torch.cuda.current_device() == runtime.local_process_index
    identity = dict(rank=rank, local_rank=runtime.local_process_index, host=socket.gethostname(),
                    device=str(runtime.accelerator.device), gpu=torch.cuda.get_device_name())
    identities = [None, None]
    torch.distributed.all_gather_object(identities, identity)
    assert {x['rank'] for x in identities} == {0,1}
    assert len({x['host'] for x in identities}) == 2, "This invocation expects two one-GPU nodes"
    value = torch.tensor(float(rank+1), device=runtime.accelerator.device)
    torch.distributed.all_reduce(value)
    assert value.item() == 3
    global_path=root/f"{phase}-global-materialization.txt"
    runtime.materialize_path(global_path, lambda p:p.write_text(str(rank)),scope="global")
    assert global_path.read_text() == "0"
    local_path=root/f"{phase}-local-materialization-{rank}.txt"
    runtime.materialize_path(local_path,lambda p:p.write_text(str(rank)),scope="local")
    assert local_path.read_text() == str(rank)
    try:
        require_single_process("offline merge")
    except RuntimeError:
        guard = True
    else: raise AssertionError("Distributed offline merge guard did not reject")
    import llm_training.trainer as trainer_module
    original_builder = trainer_module.build_training_optimizer
    coverage = {}
    def build(model, conf):
        opt = original_builder(model, conf)
        expected={id(p) for p in model.parameters() if p.requires_grad}
        actual=[id(p) for g in opt.param_groups for p in g['params']]
        assert set(actual)==expected and len(actual)==len(set(actual))
        coverage.update(expected=len(expected),actual=len(actual))
        return opt
    with runtime.model_loading_context():
        base = AutoModelForCausalLM.from_pretrained(root/'base')
    assert all(not hasattr(p, 'ds_id') for p in base.parameters())
    with patch.object(trainer_module,'build_training_optimizer',build):
        trainer=Probe(model=base,
            train_dataset=Rows(case.startswith('tt')),collate_fn=single.collate,config=cfg,runtime=runtime)
    trainer.root=root
    assert trainer.runtime is runtime
    model=runtime.accelerator.unwrap_model(trainer.model)
    shards=[]
    if cfg.deepspeed:
        assert trainer.model.zero_optimization_stage()==3 and trainer.model.bfloat16_enabled()
        shards=[dict(name=n,full=int(p.ds_numel),partition=int(p.ds_tensor.numel()))
                for n,p in model.named_parameters() if hasattr(p,'ds_tensor')]
        assert shards and any(x['partition']<x['full'] for x in shards)
    if cfg.gradient_checkpointing.enabled:
        assert model.is_gradient_checkpointing and model.config.use_cache is False
    trainables={n for n,p in model.named_parameters() if p.requires_grad}
    if cfg.lora: assert all('lora_' in n for n in trainables)
    if cfg.trainable_tokens: assert all('trainable_tokens_delta' in n for n in trainables)
    if phase=='resume':
        assert (trainer.state.global_step,trainer.state.epoch,trainer.state.batch_in_epoch)==(1,0,2)
        single.assert_rng(single.rng(),single.load(root/f'checkpoint-rng-{rank}.pt'))
        saved=single.load(root/f'checkpoint-params-{rank}.pt')
        for n,p in single.snapshot(trainer).items():torch.testing.assert_close(p,saved[n],atol=0,rtol=0)
        assert trainer.scheduler.state_dict()==single.load(root/f'checkpoint-scheduler-{rank}.pt')
    before=single.snapshot(trainer)
    token_before=effective_rows(trainer) if cfg.trainable_tokens else None
    actual=single.optimizer_chain(trainer.optimizer)[-1]
    raw=getattr(trainer.scheduler,'scheduler',trainer.scheduler)
    updates=1 if phase=='resume' else 0
    trace=[]; batches=[]
    def pre(opt,args,kwargs):trace.append(dict(event='optimizer_begin',lr=[g['lr'] for g in opt.param_groups]))
    def post(opt,args,kwargs):
        nonlocal updates
        updates+=1
        trace.append(dict(event='optimizer_end',updates=updates))
    h1=actual.register_step_pre_hook(pre);h2=actual.register_step_post_hook(post)
    original_step=trainer.scheduler.step
    def schedule(*args,**kwargs):
        old=raw.last_epoch
        ret=original_step(*args,**kwargs)
        if raw.last_epoch!=old:
            assert raw.last_epoch == updates*2, (raw.last_epoch,updates)
            assert updates>old//2
            assert all(abs(g['lr']-0.01*(1-updates/2))<1e-10 for g in actual.param_groups)
        trace.append(dict(event='scheduler_end',updates=updates,last_epoch=raw.last_epoch,lr=raw.get_last_lr()))
        return ret
    trainer.scheduler.step=schedule
    def seen(module,args,kwargs):
        if module.training:batches.extend(kwargs['input_ids'][:,1].detach().cpu().tolist())
    hook=trainer.model.register_forward_pre_hook(seen,with_kwargs=True)
    trainer.stop=phase=='interrupt'
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            try:
                trainer.train()
                assert phase!='interrupt'
            except Stop:assert phase=='interrupt'
        messages=[str(w.message) for w in captured]
    finally:
        h1.remove();h2.remove();hook.remove()
        write(root/f'{phase}-trace-rank-{rank}.json',dict(trace=trace,batches=batches))
    after=single.snapshot(trainer)
    assert any(not torch.equal(before[n],after[n]) for n in trainables)
    for n in set(before)-trainables:assert torch.equal(before[n],after[n]),n
    rows=assert_token_rows(token_before,effective_rows(trainer),(32,33)) if cfg.trainable_tokens else {}
    digest=hashlib.sha256(b''.join(after[n].contiguous().view(torch.uint8).numpy().tobytes() for n in sorted(after))).hexdigest()
    digests=[None,None];torch.distributed.all_gather_object(digests,digest)
    assert len(set(digests))==1
    artifact_probe={}
    if phase!='interrupt':
        assert (trainer.state.global_step,trainer.state.epoch,trainer.state.batch_in_epoch)==(2,1,0)
        assert updates==2 and raw.last_epoch==4 and all(g['lr']==0 for g in actual.param_groups)
        single.save(root/f'{phase}-params-{rank}.pt',after)
        trainer.model.eval()
        x=torch.tensor([[1,3,32,2]] if cfg.trainable_tokens else [[1,3,12,2]],device=runtime.accelerator.device)
        with torch.no_grad():logits=trainer.model(input_ids=x).logits.detach().float().cpu()
        assert torch.isfinite(logits).all()
        single.save(root/f'{phase}-logits-{rank}.pt',logits)
        if cfg.lora or cfg.trainable_tokens:
            import llm_training.artifacts as artifacts
            original_gather=artifacts._zero3_gather_context
            def gather(parameters,conf):
                assert parameters and all(p.requires_grad for p in parameters), 'Adapter save gathered frozen base weights'
                artifact_probe['gathered_parameters']=len(parameters)
                return original_gather(parameters,conf)
            with patch.object(artifacts,'_zero3_gather_context',gather), patch.object(model,'save_pretrained',wraps=model.save_pretrained) as save_call:
                path=trainer.save_adapter()
                assert save_call.call_count==(1 if rank==0 else 0)
                artifact_probe['save_pretrained_calls']=save_call.call_count
            if rank==0:
                from safetensors.torch import load_file
                state=load_file(str(path/'adapter_model.safetensors'))
                marker='lora_' if cfg.lora else 'trainable_tokens_delta'
                assert state and all(marker in k for k in state)
                assert not (path/'model.safetensors').exists()
        else:trainer.export_model()
    return dict(identity=identity,identities=identities,backend='nccl',world_size=2,
        state=vars(trainer.state),coverage=coverage,shards=shards,token_rows=rows,
        batches=batches,trace=trace,warnings=messages,parameter_digest=digest,
        artifact_probe=artifact_probe,offline_guard=guard)


def verify(case,root):
    for rank in range(2):
        baseline=single.load(root/f'baseline-params-{rank}.pt')
        resumed=single.load(root/f'resume-params-{rank}.pt')
        for n in baseline:torch.testing.assert_close(resumed[n],baseline[n],atol=0,rtol=0)
        torch.testing.assert_close(single.load(root/f'baseline-logits-{rank}.pt'),single.load(root/f'resume-logits-{rank}.pt'),atol=0,rtol=0)
        b=json.loads((root/f'baseline-rank-{rank}.json').read_text())['batches']
        i=json.loads((root/f'interrupt-rank-{rank}.json').read_text())['batches']
        r=json.loads((root/f'resume-rank-{rank}.json').read_text())['batches']
        assert b==i+r and len(b)==4
    ids=[json.loads((root/f'baseline-rank-{rank}.json').read_text())['batches'] for rank in range(2)]
    assert not set(ids[0])&set(ids[1]) and sorted(ids[0]+ids[1])==list(range(3,11))
    reference=single.load(root/'resume-logits-0.pt')
    tokens=case.startswith('tt');adapter=tokens or case.startswith('lora')
    fixed=torch.tensor([[1,3,32,2]] if tokens else [[1,3,12,2]],device='cuda')
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(root/'base'),root/'resumed-run/adapter').cuda().eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):logits=model(input_ids=fixed).logits.float().cpu()
        torch.testing.assert_close(logits,reference,atol=0.005,rtol=0.05)
        del model
        merge=single.merge_trainable_tokens if tokens else single.merge_lora
        merged=merge(root/'base',root/'resumed-run/adapter',root/'merged')
    else:merged=root/'resumed-run/final-model'
    model=AutoModelForCausalLM.from_pretrained(merged).cuda().eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actual=model(input_ids=fixed).logits.float().cpu()
    torch.testing.assert_close(actual,reference,atol=0.005,rtol=0.05)
    if tokens:
        original=AutoModelForCausalLM.from_pretrained(root/'base').get_input_embeddings().weight.detach()
        assert_token_rows(original,model.get_input_embeddings().weight.detach().cpu().to(original.dtype),(32,33))
    return dict(resume_parameter_equality=True,rank_parameter_equality=True,data_disjoint=True,data_complete=True,
        reload_max_abs_error=float((actual-reference).abs().max()),reload_atol=0.005,reload_rtol=0.05)


def main():
    case,phase,destination=sys.argv[1:]
    root=Path(destination)/case;root.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ.get('RANK','0'))
    result=dict(case=case,phase=phase,rank=rank,ok=False)
    try:
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        result['versions']={name:single.metadata.version(name) for name in ('torch','deepspeed','accelerate','transformers','peft')}
        result['trainer_sha256']=hashlib.sha256(Path(single.inspect.getfile(Trainer)).read_bytes()).hexdigest()
        if phase=='prepare':result.update(single.prepare(case,root))
        elif phase=='verify':result.update(verify(case,root))
        else:result.update(train(case,phase,root))
        result['ok']=True
    except Exception as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        name=f'{phase}.json' if phase in ('prepare','verify') else f'{phase}-rank-{rank}.json'
        write(root/name,result)
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
    print('RESULT '+json.dumps(result),flush=True)
    return 0 if result['ok'] else 1

if __name__=='__main__':sys.exit(main())

