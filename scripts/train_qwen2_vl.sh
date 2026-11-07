#!/usr/bin/env bash
set -euo pipefail

# From the root of the SRR repository, with LLaMA-Factory installed:
python scripts/check_srr_data.py
llamafactory-cli train configs/qwen2_vl_7b_srr_lora.yaml
