# AWS GPU smoke test

This document describes the manual AWS path for validating the single-GPU BF16 golden path in `llm-training`.

The GPU smoke test is intentionally **not** part of normal push/PR CI. Run it when a change affects CUDA execution, mixed precision, optimizer behavior on GPU, PEFT/LoRA, or another GPU-specific code path.

## What this validates

The test in `tests/gpu/test_gpu_bf16_training.py` exercises a real Hugging Face causal LM and verifies that:

- CUDA is available;
- the selected GPU supports BF16;
- Accelerate places the model on CUDA;
- the trainer uses BF16 mixed precision;
- a real forward/backward/optimizer step completes;
- model parameters change;
- a training checkpoint is written.

## Known-good AWS shape

A known-good configuration is:

- Region: `us-west-2`
- Instance: `g6.xlarge`
- GPU: NVIDIA L4 (24 GB class)
- vCPU: 4
- OS/AMI: AWS Deep Learning OSS NVIDIA Driver AMI with PyTorch 2.9 on Ubuntu 24.04, or a newer equivalent

`g5.xlarge` (NVIDIA A10G) is also suitable for this smoke test, but EC2 capacity varies by Availability Zone. If `g5.xlarge` returns `InsufficientInstanceCapacity`, try another AZ or use `g6.xlarge` rather than repeatedly retrying the same placement.

## 1. Check the EC2 GPU quota

The account needs at least 4 vCPUs in the EC2 quota named:

```text
Running On-Demand G and VT instances
```

The quota code is:

```text
L-DB2E81BA
```

Check it with:

```bash
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --region us-west-2
```

For one `g5.xlarge` or `g6.xlarge`, an applied value of `4` is sufficient.

If instance launch fails with `VcpuLimitExceeded`, increase this quota before doing anything else. The quota itself does not create an instance and does not incur compute charges.

## 2. Prepare access

Prefer AWS Systems Manager Session Manager instead of opening SSH.

The EC2 instance profile should have the managed policy:

```text
AmazonSSMManagedInstanceCore
```

Recommended instance settings:

- no inbound SSH rule is required;
- IMDSv2 required;
- root EBS volume configured with `DeleteOnTermination=true`;
- outbound network access available so the test can download its small Hugging Face model.

The repository is private, so make the source available to the instance using an authenticated mechanism such as a read-only deploy key/GitHub App token or a temporary private S3 staging object. Do not embed a long-lived GitHub token in user data, an AMI, or the repository.

## 3. Launch the GPU instance

Launch one `g6.xlarge` (or `g5.xlarge`) with the Deep Learning AMI and the SSM instance profile.

After launch, wait until both of these are true:

```text
EC2 instance state: running
SSM PingStatus: Online
```

If EC2 returns `InsufficientInstanceCapacity`, change AZ or switch between `g5.xlarge` and `g6.xlarge`. This error is a capacity problem, not a quota problem.

## 4. Activate the PyTorch environment

Recent AWS PyTorch 2.9 DLAMIs keep PyTorch and CUDA libraries in a virtual environment under `/opt/pytorch`.

Do not assume the system `python3` contains PyTorch. Activate the environment first:

```bash
source /opt/pytorch/bin/activate
```

When running through an SSM shell command that uses `/bin/sh`, `source` may fail with `source: not found`. Use Bash explicitly:

```bash
bash -lc 'source /opt/pytorch/bin/activate && python --version'
```

or use POSIX dot syntax:

```bash
. /opt/pytorch/bin/activate
```

## 5. Verify the GPU before installing the project

Run:

```bash
nvidia-smi
```

Then:

```bash
bash -lc 'source /opt/pytorch/bin/activate && python -c "import torch; print(\"torch=\", torch.__version__); print(\"cuda_available=\", torch.cuda.is_available()); print(\"device=\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print(\"bf16_supported=\", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"'
```

The important checks are:

```text
cuda_available= True
bf16_supported= True
```

If the system `python3` raises `ModuleNotFoundError: No module named 'torch'`, that normally means `/opt/pytorch` was not activated.

## 6. Install this repository

From the repository root:

```bash
bash -lc 'source /opt/pytorch/bin/activate && pip install -e ".[dev]"'
```

The DLAMI already provides the CUDA-enabled PyTorch build. `pip` should not need to replace it just to run this test.

## 7. Run only the GPU smoke test

Run:

```bash
bash -lc 'source /opt/pytorch/bin/activate && pytest -m gpu tests/gpu -q'
```

A successful run should end with output similar to:

```text
step=1 loss=...
1 passed
```

The exact loss value is not a contract; successful finite training, parameter mutation, and checkpoint creation are what matter.

The test downloads `sshleifer/tiny-gpt2`. An unauthenticated Hugging Face Hub warning is acceptable for an occasional smoke test, although a token can be configured if rate limits become a problem.

## 8. Terminate the instance

Terminate the EC2 instance immediately after the test, whether the test passes or fails.

Then verify:

```text
instance state: terminated
```

Also remove any temporary credentials, staging S3 objects/buckets, IAM policies, roles, or instance profiles that were created only for the smoke test.

Do not leave a GPU instance merely stopped unless there is a deliberate reason to retain its EBS volume. A terminated instance with a delete-on-termination root volume is the cleanest default for an ephemeral test.

## Normal CI vs GPU smoke

Normal push/PR CI must exclude GPU tests:

```bash
pytest -m "not integration and not gpu"
```

CPU Hugging Face integration tests should also exclude GPU tests:

```bash
pytest -m "integration and not gpu" tests/integration
```

The repository's `.github/workflows/gpu-smoke.yml` is intentionally manual (`workflow_dispatch`) and targets a GPU-labeled self-hosted runner. It is separate from ordinary CI so that pushes and pull requests do not create GPU cost.

## Troubleshooting summary

| Symptom | Meaning | Fix |
| --- | --- | --- |
| `VcpuLimitExceeded` | GPU-family EC2 quota is too low | Increase `Running On-Demand G and VT instances` |
| `InsufficientInstanceCapacity` | That instance/AZ has no current capacity | Try another AZ or switch `g5.xlarge` / `g6.xlarge` |
| `ModuleNotFoundError: torch` | System Python was used | Activate `/opt/pytorch` |
| `source: not found` | SSM command is running under `/bin/sh` | Use `bash -lc` or `. /opt/pytorch/bin/activate` |
| `cuda_available=False` | CUDA environment/device is not active | Check AMI, driver, instance type, and activated environment |
| `bf16_supported=False` | Selected GPU does not support the required BF16 path | Use a supported GPU such as L4/A10G |

## When to run this test

Run the AWS GPU smoke test after changes that can plausibly behave differently on CUDA, for example:

- mixed-precision logic;
- CUDA/device placement;
- LoRA/PEFT integration;
- optimizer changes that rely on fused or GPU-specific kernels;
- gradient checkpointing or compile paths;
- distributed-training work before expanding to multi-GPU tests.

For ordinary Python refactors, configuration changes, CPU-only checkpoint logic, or documentation changes, normal CI is sufficient.

## Extended PEFT and DeepSpeed validation

See the [GPU test matrix](gpu-validation-matrix.md) and
[permanent workers](../tests/gpu/README.md). The September 4 rerun passes all
8 single-L4 cases / 40 isolated phases and 70 CPU regressions on the fixed code.
Run each backend and lifecycle phase in a fresh process, require both a zero
exit status and successful assertions, and use a new output directory. Tests
must compare actual parameter updates, LR, data order, RNG, and artifact logits.

For clean CPU offload, use matching torch/CUDA versions and a fresh extension
cache with DS_SKIP_CUDA_CHECK absent. Preserve exact source, package versions,
all logs and artifacts locally before deleting temporary cloud resources.
Two-GPU tests are separate work; this suite intentionally uses only GPU 0.
