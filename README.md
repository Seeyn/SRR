# Beyond Risky Activities: Bridging the Supervision Gap for Situational Risk Reasoning

**NeurIPS 2026 · Evaluations and Datasets Track**

[🤗 **SRR Dataset (Hugging Face)**](https://huggingface.co/datasets/Anonymous-SRR/SRR) · [**Training and Reproduction**](TRAINING.md)

> **An activity can be safe in isolation but unsafe in context.** Situational Risk Reasoning (SRR) provides supervision for vision-language models (VLMs) to identify safety risks arising from the interaction between a user's intended activity and the surrounding visual scenario.

## Overview

Existing safety-alignment datasets primarily emphasize intrinsically harmful activities. SRR instead provides **activity–scenario supervision**: whether a seemingly benign activity is safe can depend on the visual environment. The dataset includes synthesized visual scenarios, user queries, and structured safety-reasoning targets for supervised fine-tuning (SFT).

Each target response follows a safety chain-of-thought format:

```text
<think> ... activity, scene, and situational risk reasoning ... </think>
<judgement> Safe / Unsafe </judgement>
<answer> ... user-facing response ... </answer>
```

The dataset covers nine sources of situational risk: **Distraction, Sport, Etiquette, Weather, Electricity, Fire, Animal, Location, and Hygiene**.

## Dataset

The [public SRR dataset](https://huggingface.co/datasets/Anonymous-SRR/SRR) contains **6,169 training examples**, each with an image and a user–assistant conversation.

| Data source | Safe examples | Unsafe examples | Total |
| --- | ---: | ---: | ---: |
| Core SRR | 873 | 1,319 | 2,192 |
| VLGuard | 977 | 2,000 | 2,977 |
| LLaVA-1.5 clean subset | 1,000 | 0 | 1,000 |
| **Total** | **2,850** | **3,319** | **6,169** |

The incorporated examples are reformatted to use the structured SRR target-response format, as described in the paper. Please consult the [dataset card](https://huggingface.co/datasets/Anonymous-SRR/SRR) for the data and licensing terms.

### Load the dataset

```python
from datasets import load_dataset

dataset = load_dataset("Anonymous-SRR/SRR", split="train")
print(len(dataset))  # 6169
print(dataset[0]["messages"])
image = dataset[0]["image"]
```

The Hugging Face dataset stores images directly. The training preparation script below exports them into local image files and generates a ShareGPT-style JSONL file for LLaMA-Factory, avoiding machine-specific absolute paths.

## Training and Reproduction

We provide a **Qwen2-VL-7B-Instruct LoRA fine-tuning recipe** based on LLaMA-Factory. 

### 1. Prepare the data

From the repository root:

```bash
pip install datasets pillow
python scripts/prepare_srr.py
python scripts/check_srr_data.py
```

This produces `data/srr_train.jsonl` and `data/images/`. The exported dataset is registered as `srr_train` in `data/dataset_info.json`.

### 2. Train Qwen2-VL-7B

Install a compatible [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) environment first. The original training environment used `llamafactory 0.9.5.dev0`; see [TRAINING.md](TRAINING.md) for configuration details.

For **two GPUs**, matching the effective batch size of 32 reported in the paper:

```bash
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 \
  llamafactory-cli train configs/qwen2_vl_7b_srr_lora.yaml
```

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen2-VL-7B-Instruct` |
| Fine-tuning | LoRA SFT; `qwen2_vl` template |
| LoRA rank / alpha / dropout | `8` / `16` / `0.0` |
| LoRA targets | Attention and MLP projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) |
| Learning rate / epochs | `2e-4` / `3` |
| Per-device batch / accumulation | `2` / `8` |
| Global effective batch (2 GPUs) | `32` |
| Precision / scheduler | `bf16` / cosine (warmup ratio `0.1`) |

LoRA values and target modules were verified against the trained adapter's `adapter_config.json`. For complete settings, data-format details, and limitations of exact reproducibility, refer to **[TRAINING.md](TRAINING.md)** and `configs/qwen2_vl_7b_srr_lora.yaml`.


**Release scope:** This repository provides the data preparation and Qwen2-VL LoRA training recipe. A downloadable fine-tuned checkpoint and standalone evaluation runners are **not included in this release**. The reported results should not be interpreted as guarantees for real-world safety-critical use.

## Responsible Use and Limitations

SRR is intended for research on VLM post-training and situational safety reasoning. Its synthetic images and automatically generated reasoning targets may contain imperfections. Human validation was conducted on a sample rather than every training example. Benchmark performance does not establish suitability for safety-critical real-world deployment.

## Citation

If you use SRR in your research, please cite:

```bibtex
@inproceedings{li2026beyond,
  title     = {Beyond Risky Activities: Bridging the Supervision Gap for Situational Risk Reasoning},
  author    = {Li, Zelin and Liu, Yifan and Yao, Ruichen and Liu, Yaokun and Gupta, Rahul and Wang, Dong},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {Evaluations and Datasets Track}
}
```

## License

The dataset on Hugging Face is released under **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**. Review the [dataset card](https://huggingface.co/datasets/Anonymous-SRR/SRR) and any applicable upstream usage conditions before reuse. No separate code license has yet been specified for the GitHub training scripts.
