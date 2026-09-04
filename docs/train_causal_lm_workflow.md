# Portable causal-LM training workflow

`run_training_workflow` composes local data preparation, optional vocabulary
expansion, training, export/merge, and optional fresh-process generation using
ordinary Python calls and paths. Install the companion `llm-training-data`
package for its actual formatter/tokenizer/masking collator. Core `Trainer` does
not require that package.

## Usage

```python
from llm_training import (
    EvaluationConfig, LoRAConfig, TrainingConfig, run_training_workflow,
)

result = run_training_workflow(
    base_path="models/base",  # existing local HF model + tokenizer
    train_data="data/train.jsonl",
    validation_data="data/validation.jsonl",
    test_data="data/test.jsonl",
    config=TrainingConfig(
        output_dir="runs/example",  # new directory
        mixed_precision="bf16",
        lora=LoRAConfig(),
        evaluation=EvaluationConfig(every_epochs=1),
    ),
    inference=True,
    inference_device="cuda",
    max_sequence_length=512,
    max_new_tokens=32,
)
print(result.model_path, result.predictions_path, result.evaluation)
```

Each JSONL row requires string `prompt` and `completion` fields; an optional `id`
can identify samples. Lists of dictionaries are also accepted. Train, validation
and test inputs are explicit. Cross-split overlap is rejected using supplied IDs
and exact prompt/completion pairs; this does not detect semantic duplicates.
Inference scores the test split when present, otherwise the validation split.
Missing scoring data fails before training. The driver never constructs a test
split from the training data.

The first driver supports one training process, optionally with single-GPU
DeepSpeed. Use `DistributedRuntime`/`Trainer` directly and the dedicated launchers
for multi-rank jobs. This keeps deployment/topology decisions with the caller.
The driver requires a new output directory and runs on POSIX systems.

## Lifecycle

```text
local base + tokenizer -> optional vocabulary expansion
                                  |
explicit train/validation -> real SFT collator -> training process
                                  |
                    checkpoint + full model or adapter
                                  |
                   separate offline merge process (PEFT)
                                  |
                        manifest verification
                                  |
                separate model reload / generation process
                                  |
                       predictions + exact match
```

Training, offline merge and inference each own their model in a fresh process.
A failed or timed-out stage prevents downstream execution; stage logs and JSON
requests/results remain under `workflow/`. The driver waits for training to exit
before merging. System prompt settings use the data package's shared formatter
in both training and generation. Generation is greedy and returns explicit
prediction records with expected completions. Task-specific parsing, retrieval,
judges and serving engines remain application responsibilities.

`WorkflowResult` names the final model, training checkpoint, optional adapter,
optional predictions file and validation/generation results. Training validation
uses [global token-weighted evaluation](evaluation.md). Generation currently
reports example count and exact match after trimming surrounding whitespace.

## Vocabulary expansion

```python
from llm_training import expand_vocabulary, TrainableTokensConfig

expanded = expand_vocabulary(
    "models/base", "models/expanded", ["<topic_a>", "<topic_b>"],
    initialization="mean", seed=42,
)
if expanded.n_tokens_added:
    adapter_config = TrainableTokensConfig(n_tokens_added=expanded.n_tokens_added)
```

Existing/duplicate tokens do not count as additions. The utility records actual
contiguous appended IDs and saves model/tokenizer together. Existing input and
output rows are copied exactly. For untied tables, each table is handled
separately. `mean` initializes new rows from the old-row mean; `random` uses a
seeded normal distribution at the old table's standard deviation.

Input/output embedding counts must match the tokenizer before expansion; models
with intentionally padded vocabulary tables need a separate policy. Use
`new_tokens=[...]` on the workflow to perform this stage automatically.
`TrainableTokensConfig.n_tokens_added` must match actual additions (or an existing
expanded-base manifest). Zero additions cannot request TrainableTokens training.
Standalone LoRA and TrainableTokens remain mutually exclusive.

## Artifact identity and resume

Workflow base/checkpoint/adapter/model directories include `artifact.json` with:

- artifact kind and manifest version;
- exact base weight/config fingerprint and informational base path;
- tokenizer vocabulary/special-token/chat-template fingerprint;
- actual added token strings and IDs;
- training and generation configuration;
- SHA-256 inventory of the artifact files.

`ArtifactManifest.load(path).validate(path, base_path=...)` verifies file integrity,
tokenizer identity, token IDs and the expected base before merge/resume. Paths in
the manifest are informational; moved artifacts can be validated against a new
explicit base location. Full-model inference validates its artifact without
requiring the original base directory.

Pass an absolute workflow checkpoint path via `CheckpointConfig.resume_from` to
resume into a new run directory with the same base/configuration. The workflow
requires its artifact manifest for this path; lower-level `Trainer` also supports
legacy checkpoints. A checkpoint restores optimizer/scheduler/RNG/trainer state;
an adapter or exported model is not a training resume checkpoint.

The manifest fingerprints bytes and tokenizer semantics, not publisher trust.
The utility assumes standard local Hugging Face weight/config files. Core
`Trainer` export APIs remain independently usable; callers outside this workflow
can explicitly call `write_artifact_manifest`.

## Scope

Data preprocessing, external storage transfer, scheduling, experiment tracking,
and domain-specific business flows stay outside this implementation. No remote
platform is required. See [tests](tests.md) and [roadmap](roadmap.md) for validation
coverage and deferred extensions.
