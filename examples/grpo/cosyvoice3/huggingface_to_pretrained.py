#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert HuggingFace GRPO output back into CosyVoice3 llm.pt format.

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer
from safetensors import safe_open


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-cosyvoice3-llm-path",
        type=str,
        required=True,
        help="HuggingFace model directory produced by GRPO.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./llm.pt",
        help="Path to write CosyVoice3 llm.pt.",
    )
    return parser.parse_args()


def _infer_speech_token_size(tokenizer: AutoTokenizer) -> int:
    max_idx = -1
    for token in tokenizer.get_vocab().keys():
        if token.startswith("<|s_") and token.endswith("|>"):
            match = re.match(r"<\|s_(\d+)\|>", token)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
    if max_idx < 0:
        raise ValueError("Failed to infer speech token size from tokenizer.")
    return max_idx + 1


def _load_hf_tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        tensors = {}
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        return tensors
    bin_path = model_dir / "pytorch_model.bin"
    if not bin_path.exists():
        raise FileNotFoundError("No model.safetensors or pytorch_model.bin found.")
    return torch.load(bin_path, map_location="cpu")


def main():
    args = get_args()
    model_dir = Path(args.hf_cosyvoice3_llm_path)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_cosyvoice3_llm_path)

    speech_start_idx = tokenizer.convert_tokens_to_ids("<|s_0|>")
    if speech_start_idx is None or speech_start_idx < 0:
        raise ValueError("Missing <|s_0|> token in tokenizer.")

    speech_token_size = _infer_speech_token_size(tokenizer)
    speech_total = speech_token_size + 200

    hf_tensors = _load_hf_tensors(model_dir)
    cosy_tensors = {}

    for k, tensor in hf_tensors.items():
        if k == "lm_head.bias":
            # Skip bias used to mask non-speech tokens.
            continue
        cosy_key = "llm.model." + k
        cosy_tensors[cosy_key] = tensor
        if k.startswith("lm_head"):
            cosy_tensors["llm_decoder.weight"] = tensor[
                speech_start_idx : speech_start_idx + speech_total
            ]
        if k.startswith("model.embed_tokens"):
            cosy_tensors["speech_embedding.weight"] = tensor[
                speech_start_idx : speech_start_idx + speech_total
            ]

    # Original CosyVoice3 expects text vocab size of 151936 (not speech_start_idx)
    # The HF model has extended vocab: text (151936) + speech tokens (6784)
    original_text_vocab_size = 151936
    cosy_tensors["llm.model.model.embed_tokens.weight"] = cosy_tensors[
        "llm.model.model.embed_tokens.weight"
    ][:original_text_vocab_size]
    cosy_tensors["llm.model.lm_head.weight"] = cosy_tensors[
        "llm.model.model.embed_tokens.weight"
    ]

    torch.save(cosy_tensors, args.output_path)


if __name__ == "__main__":
    main()
