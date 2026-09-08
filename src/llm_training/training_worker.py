"""Subprocess entry point for high-level training stages."""

import json
import os
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .artifacts import merge_lora, merge_trainable_tokens
from .checkpoint import find_latest_checkpoint
from .data import adapt_column_batch_collator
from .distributed import DistributedRuntime
from .inspection import build_training_plan, format_training_plan, save_training_plan
from .manifest import ArtifactManifest, write_artifact_manifest
from .run import RunIdentity, write_run_identity
from .trainer import Trainer
from .training import training_config_from_dict


def run(stage, request):
    config = training_config_from_dict(request["config"])
    root = Path(config.output_dir)
    base = request["base_path"]
    identity = RunIdentity(**request["run_identity"])
    write_run_identity(identity)
    kwargs = dict(
        base_path=base,
        new_tokens=request["new_tokens"],
        training_config=config.to_dict(),
        generation_config={
            "system_prompt": request["system_prompt"],
            "max_new_tokens": request["max_new_tokens"],
        },
    )
    if stage == "train":
        from llm_training_data import build_sft_collator

        runtime = DistributedRuntime(config)
        components = runtime.load_causal_lm(
            base,
            model_kwargs={"local_files_only": True},
            tokenizer_kwargs={"local_files_only": True},
        )
        collate = adapt_column_batch_collator(
            build_sft_collator(
                components.tokenizer,
                max_sequence_length=request["max_sequence_length"],
                pad_to_multiple_of=1,
                system_prompt=request["system_prompt"],
            )
        )
        trainer = Trainer(
            model=components.model,
            tokenizer=components.tokenizer,
            runtime=runtime,
            train_dataset=request["splits"]["train"],
            eval_dataset=request["splits"]["validation"],
            collate_fn=collate,
            config=config,
        )
        plan = build_training_plan(
            config,
            train_examples=len(request["splits"]["train"]),
            world_size=runtime.num_processes,
            adaptation_summary=trainer.adaptation_summary,
            model_name=str(base),
            max_sequence_length=request["max_sequence_length"],
            has_eval_data=request["splits"]["validation"] is not None,
            run_identity=identity,
        )
        trainer.accelerator.print(format_training_plan(plan))
        if trainer.accelerator.is_main_process:
            save_training_plan(plan, root)
        trainer.accelerator.wait_for_everyone()
        trainer.train()
        checkpoint = find_latest_checkpoint(root)
        write_artifact_manifest(
            checkpoint, kind="checkpoint", tokenizer=components.tokenizer, **kwargs
        )
        if config.lora is not None or config.trainable_tokens is not None:
            adapter = trainer.save_adapter(root / "artifacts")
            write_artifact_manifest(
                adapter, kind="adapter", tokenizer=components.tokenizer, **kwargs
            )
            model_path = None
        else:
            trainer.export_model(root / "artifacts")
            model_path = root / "artifacts" / "final-model"
            write_artifact_manifest(
                model_path, kind="model", tokenizer=components.tokenizer, **kwargs
            )
            adapter = None
        return dict(
            checkpoint_path=str(checkpoint),
            adapter_path=str(adapter) if adapter else None,
            model_path=str(model_path) if model_path else None,
            evaluation=trainer.evaluation_history,
            device=str(runtime.accelerator.device),
            world_size=runtime.num_processes,
            mixed_precision=config.mixed_precision,
            run_id=identity.run_id,
            artifact_uri=identity.artifact_uri,
        )

    if stage == "merge":
        adapter = request["adapter_path"]
        ArtifactManifest.load(adapter).validate(adapter, base_path=base)
        destination = root / "model"
        merge = merge_lora if config.lora is not None else merge_trainable_tokens
        merge(base, adapter, destination)
        tokenizer = AutoTokenizer.from_pretrained(destination, local_files_only=True)
        write_artifact_manifest(
            destination, kind="model", tokenizer=tokenizer, **kwargs
        )
        return dict(model_path=str(destination))

    if stage == "infer":
        from llm_training_data import ChatPromptFormatter

        path = request["model_path"]
        ArtifactManifest.load(path).validate(path)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = (
            AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
            .to(request["inference_device"])
            .eval()
        )
        formatter = ChatPromptFormatter(tokenizer, request["system_prompt"])
        predictions = []
        with torch.no_grad():
            for row in request["score_data"]:
                prompt = formatter(row["prompt"])
                inputs = tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=False
                ).to(model.device)
                if inputs.input_ids.numel() == 0:
                    raise ValueError("Inference prompt tokenizes to an empty sequence")
                output = model.generate(
                    **inputs,
                    max_new_tokens=request["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                text = tokenizer.decode(
                    output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
                )
                predictions.append(
                    dict(
                        prompt=row["prompt"],
                        expected=row["completion"],
                        prediction=text,
                    )
                )
        output_path = root / "predictions.jsonl"
        output_path.write_text("".join(json.dumps(row) + "\n" for row in predictions))
        metrics = {
            "examples": len(predictions),
            "exact_match": sum(
                p["prediction"].strip() == p["expected"].strip() for p in predictions
            )
            / len(predictions),
        }
        return dict(predictions_path=str(output_path), metrics=metrics)

    raise ValueError(f"Unknown training stage: {stage}")


if __name__ == "__main__":
    stage, request_path, result_path = sys.argv[1:]
    result = run(stage, json.loads(Path(request_path).read_text()))
    Path(result_path).write_text(
        json.dumps(dict(ok=True, pid=os.getpid(), **result), indent=2) + "\n"
    )
