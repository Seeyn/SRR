# Reproducing SRR Fine-Tuning (Qwen2-VL-7B)

This recipe documents the **user-provided LLaMA-Factory SFT configuration** for Qwen2-VL-7B-Instruct, with portable data paths. The 6,169-example SRR dataset is hosted at [Anonymous-SRR/SRR](https://huggingface.co/datasets/Anonymous-SRR/SRR). Large images, JSONL exports, and trained checkpoints are **not** committed to GitHub.

## Directory structure

```text
SRR/
├── README.md
├── TRAINING.md
├── .gitignore
├── configs/
│   └── qwen2_vl_7b_srr_lora.yaml
├── data/
│   ├── dataset_info.json             # committed
│   ├── srr_train.jsonl               # generated; gitignored
│   └── images/                       # generated; gitignored
└── scripts/
    ├── prepare_srr.py
    ├── check_srr_data.py
    └── train_qwen2_vl.sh
```

## 1. Install LLaMA-Factory

Follow the [official installation instructions](https://github.com/hiyouga/LlamaFactory). The installed version of LLaMA-Factory, transformers, torch, PEFT, CUDA and Python should be documented for strict reproducibility. The historical version used for the paper was not supplied in the available settings.

For data export only:

```bash
pip install datasets pillow
```

## 2. Prepare local data

From the **SRR repository root**:

```bash
python scripts/prepare_srr.py
python scripts/check_srr_data.py
```

The exporter downloads the public HF dataset, writes all messages into `data/srr_train.jsonl`, and stores images under `data/images/` using lossless PNG encoding and content-derived names. The relative path in each JSONL row is `images/<hash>.png`, which is resolved from `dataset_dir: data`.

Small smoke test (not for full training):

```bash
python scripts/prepare_srr.py --limit 10
python scripts/check_srr_data.py --expected-count 10
```

Run the full exporter again before training to replace the 10-line JSONL.

Data-source and conversion note: The attached original training JSONL uses source-machine absolute paths (e.g., COCO, VLGuard and SRR-generated image directories). These are deliberately **not** distributed. The public HF dataset has 6,169 rows with `image` and `messages` fields. To claim *bit-for-bit identical* training data, compare the HF version against the original training dataset and original image files, particularly record ordering and image decoding.

## 3. Fine-tune

From the SRR repository root, with LLaMA-Factory installed:

```bash
bash scripts/train_qwen2_vl.sh
```

Or directly:

```bash
llamafactory-cli train configs/qwen2_vl_7b_srr_lora.yaml
```

Key hyperparameters preserved from the supplied YAML:

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen2-VL-7B-Instruct` |
| LLaMA-Factory template | `qwen2_vl` |
| Stage / fine-tuning | `sft` / `lora` |
| Learning rate | `2e-4` |
| Cutoff length | `2048` |
| Batch size per device | `2` |
| Gradient accumulation | `8` |
| Epochs | `3` |
| Scheduler / warmup | `cosine` / `0.1` |
| Precision | `bf16` |
| Evaluation | Not enabled in the provided YAML |

**Important missing metadata to verify:**

- The original YAML specifies `dataset: sinr_woobsthink`, while the provided dataset registry entry is named `sinr_woobs`. The public recipe consistently renames both to `srr_train`. Check that the released HF dataset contains exactly the examples used by the actual original experiment.
- LoRA rank, alpha, dropout and target modules were **not set explicitly** in the supplied YAML. They may depend on the historical LLaMA-Factory defaults or separate arguments. Do not assume values without verification.
- The effective batch size is `per_device_train_batch_size * gradient_accumulation_steps * number_of_processes`, which is `16 × number of processes` for this recipe (32 on 2 processes/GPUs).
- For strict reproducibility, add seed, GPU count, framework commit and package/CUDA versions from the original run.
- This package preserves the supplied training settings; it has not been run through a full GPU training or independent benchmark reproduction.

## Responsible use

The publicly released SRR dataset is licensed under CC BY-NC 4.0. Refer to the dataset card and upstream source licensing/usage conditions. This recipe is for research reproducibility; performance on SRR evaluation benchmarks does not ensure safety-critical deployment readiness.
