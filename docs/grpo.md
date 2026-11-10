# Config-driven GRPO training

`llm-training` supports Group Relative Policy Optimization through the same
`llm-train --config` entry point used by SFT, causal-LM, and DPO jobs. Set
`data.format: grpo`; the runtime uses TRL's `GRPOTrainer` while reusing the
package's model loading, LoRA/freeze policy, precision, DeepSpeed, checkpoint,
evaluation, and artifact settings.

The GRPO surface remains task-neutral. Domain-specific constraints, remote
reward services, and application orchestration remain outside the package.

## Install

For full-model GRPO:

```bash
pip install "llm-training[grpo]"
```

For LoRA:

```bash
pip install "llm-training[grpo,peft]"
```

Add the vLLM integration for either vLLM rollout backend:

```bash
pip install "llm-training[grpo,vllm]"
```

Add DeepSpeed when using ZeRO:

```bash
pip install "llm-training[grpo,peft,deepspeed]"
```

DPO and GRPO use the same supported TRL release. Install
`llm-training[rl]` when both trainers are needed in one environment.

## Prompt data

Input is local JSONL. Each non-empty row must contain a prompt. Other columns
are preserved and forwarded to reward functions.

```json
{"prompt":"What is 2 + 2? Return only the number.","answer":"4","row_id":"example-1"}
{"prompt":"What is 3 + 5? Return only the number.","answer":"8","row_id":"example-2"}
```

The prompt column is configurable:

```yaml
data:
  train_file: data/prompts.jsonl
  validation_file: data/validation.jsonl
  format: grpo
  prompt_field: question
```

A `system_prompt` can be configured under `grpo`. It is rendered through
the tokenizer's chat template before generation.

## Configuration

A complete LoRA example is in
[`examples/train_grpo.yaml`](../examples/train_grpo.yaml).

```yaml
grpo:
  rollout:
    backend: transformers
  num_generations: 8
  max_completion_length: 256
  temperature: 1.0
  top_p: 1.0
  beta: 0.0
  epsilon: 0.2
  epsilon_high: null
  loss_type: dapo
  scale_rewards: group
  num_iterations: 1
  steps_per_generation: null
  mask_truncated_completions: true
  rewards:
    - module_path: llm_training.rewards.exact_match_reward
      weight: 1.0
      abstain: error
      kwargs:
        strip: true
```

The effective number of generated rows per device is
`per_device_batch_size * steps_per_generation`. When
`steps_per_generation` is omitted it uses
`gradient_accumulation_steps`. That number must be divisible by
`num_generations`, so every prompt group remains complete.

The supported loss normalization modes are `dapo`, `grpo`, `bnpo`, and
`dr_grpo`. Reward scaling may be `group`, `batch`, or `none`.

## Rollout backends

`grpo.rollout.backend` has three explicit values:

| Backend | Behavior |
| --- | --- |
| `transformers` | Uses the policy model's Transformers generation path. This is the default. |
| `vllm_colocate` | Runs vLLM in the training process and shares the training GPU resources. |
| `vllm_server` | Connects the trainer to a separately launched vLLM server. |

Colocated vLLM example:

```yaml
grpo:
  rollout:
    backend: vllm_colocate
    gpu_memory_utilization: 0.3
    max_model_length: 4096
    tensor_parallel_size: 1
    enable_sleep_mode: true
```

Server example:

```yaml
grpo:
  rollout:
    backend: vllm_server
    server_base_url: http://127.0.0.1:8000
    server_timeout: 240
```

The server is a separate process and must be running before training starts.
Its GPUs must not overlap with the trainer's GPUs. `server_base_url` is
optional; when omitted, TRL's server defaults are used. GPU memory, model
length, tensor parallelism, and sleep mode are configured when launching the
server; the matching `rollout` fields apply only to `vllm_colocate`.

## Constrained decoding

vLLM rollouts support task-neutral regex constrained decoding through the
same config file:

```yaml
grpo:
  rollout:
    backend: vllm_colocate
  constraint:
    kind: regex
    pattern: '[0-9]+'
```

The pattern is passed unchanged to TRL's vLLM structured-output integration.
It constrains generated completions; it does not alter reward calculation or
training labels. Constraints are rejected with the `transformers` backend so
a config cannot appear valid while silently running unconstrained generation.
The current public contract supports `kind: regex`; JSON Schema and grammar
constraints are not accepted as aliases.

## Reward plug-ins

Each reward is a dotted path to a callable function or class. Functions receive
TRL's keyword arguments, including `prompts`, `completions`, and every
additional dataset column. They return one finite number per completion.

```python
def correctness_reward(*, completions, answer, **kwargs):
    return [
        float(completion.strip() == expected.strip())
        for completion, expected in zip(completions, answer)
    ]
```

Configuration-time `kwargs` are bound to the function. If the target is a
class, the class is constructed with those arguments and the resulting callable
is used as the reward.

Multiple rewards are passed to TRL in configuration order and combined with
their declared weights.

### Explicit abstention behavior

A reward returning `None` is ambiguous during multi-reward aggregation. This
runtime therefore requires an explicit policy:

- `abstain: error` is the default and fails immediately.
- `abstain: zero` converts an abstention to `0.0`.

Non-finite values and non-scalar tensor values always fail. This prevents
missing reward values from silently changing group advantages.

Two small task-neutral rewards are bundled:

- `llm_training.rewards.exact_match_reward`
- `llm_training.rewards.RegexReward`

They are suitable for examples and smoke tests; production jobs can provide
their own importable reward package.

## Validate and run

Validate the config and all JSONL rows without loading model weights:

```bash
llm-train --config examples/train_grpo.yaml --dry-run
```

Run one process:

```bash
llm-train --config examples/train_grpo.yaml
```

Run multiple GPUs:

```bash
accelerate launch --num_processes 8 -m llm_training.cli \
  --config examples/train_grpo.yaml
```

The dry run rejects unknown fields, missing prompts, invalid reward definitions,
and incomplete generation groups.

## Evaluation and checkpoints

When a validation file is present, `evaluation.every_steps` enables
step-based evaluation. Otherwise `evaluation.every_epochs: 1` evaluates once
per epoch. Without validation data, evaluation is disabled.

By default, `checkpoint.save_every_steps: null` writes only the final model
or adapter. A positive cadence enables intermediate TRL checkpoints and
`checkpoint.keep_last` controls retention.

The final artifact includes `grpo_config.json`, tokenizer files, and an
optional `system_prompt.txt`.

## Scope

Included:

- callable reward plug-ins and weighted multi-reward aggregation;
- full-model, selective, and LoRA training;
- Accelerate and DeepSpeed ZeRO;
- checkpoint/resume, validation, and final artifact export;
- JSON/YAML configuration and dry-run validation.

Deferred:

- JSON Schema or grammar constrained decoding;
- token-level credit assignment and auxiliary losses;
- dynamic resampling;
- remote reward-model execution;
- task-specific evaluation pipelines.

Those features require explicit rollout and trainer-extension contracts rather
than hidden modifications to the base trainer.
