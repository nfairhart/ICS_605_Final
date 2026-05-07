#!/usr/bin/env python3
"""
Convert saved LoRA adapter -> GGUF for LM Studio (runs locally on M1 Mac).

Setup (one-time):
  pip install transformers peft torch accelerate sentencepiece
  huggingface-cli login        # Gemma is a gated model — accept license at hf.co/google/gemma-4-e2b-it
  brew install llama.cpp       # for Q4_K_M quantization (optional but recommended)

Run:
  python convert_lora_to_gguf.py
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

LORA_DIR   = "gemma4-e2b-resume-lora"   # unzipped LoRA adapter directory
BASE_MODEL = "google/gemma-4-e2b-it"
MERGED_DIR = "gemma4-e2b-resume-merged"
GGUF_DIR   = "gemma4-e2b-resume-gguf"
QUANT      = "Q4_K_M"


def step_merge():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"Loading {BASE_MODEL} in fp16 on CPU (~4 GB, may take a few minutes)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print(f"Loading LoRA adapter from {LORA_DIR}...")
    model = PeftModel.from_pretrained(model, LORA_DIR)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_DIR}/...")
    model.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"Merge complete -> {MERGED_DIR}/")


def step_clone_llama():
    llama = Path("llama.cpp")
    convert_script = llama / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print("Cloning llama.cpp (conversion scripts only, no compilation needed)...")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/ggerganov/llama.cpp", str(llama)],
            check=True,
        )
    else:
        print("llama.cpp already present.")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "gguf", "sentencepiece"],
        check=True,
    )
    return llama


def step_convert(llama):
    os.makedirs(GGUF_DIR, exist_ok=True)
    f16_path = f"{GGUF_DIR}/gemma4-e2b-resume.f16.gguf"
    print(f"Converting to F16 GGUF -> {f16_path}...")
    subprocess.run(
        [sys.executable, str(llama / "convert_hf_to_gguf.py"),
         MERGED_DIR, "--outfile", f16_path, "--outtype", "f16"],
        check=True,
    )
    size_gb = os.path.getsize(f16_path) / 1e9
    print(f"F16 GGUF saved ({size_gb:.1f} GB)")
    return f16_path


def step_quantize(f16_path):
    quantize_bin = shutil.which("llama-quantize")
    if not quantize_bin:
        print(
            "\nllama-quantize not found — skipping quantization.\n"
            f"You can load {f16_path} (~3.5 GB) directly in LM Studio.\n"
            "For the smaller Q4_K_M file (~1.2 GB): brew install llama.cpp and re-run."
        )
        return f16_path

    q_path = f"{GGUF_DIR}/gemma4-e2b-resume.Q4_K_M.gguf"
    print(f"Quantizing to {QUANT} -> {q_path}...")
    subprocess.run([quantize_bin, f16_path, q_path, QUANT], check=True)
    size_gb = os.path.getsize(q_path) / 1e9
    print(f"Q4_K_M GGUF saved ({size_gb:.1f} GB)")
    return q_path


if __name__ == "__main__":
    if not Path(LORA_DIR).exists():
        print(f"ERROR: {LORA_DIR}/ not found. Unzip your downloaded LoRA zip here first.")
        sys.exit(1)

    print("=== Step 1: Merge LoRA into base model ===")
    step_merge()

    print("\n=== Step 2: Get llama.cpp conversion script ===")
    llama = step_clone_llama()

    print("\n=== Step 3: Convert merged model to F16 GGUF ===")
    f16_path = step_convert(llama)

    print("\n=== Step 4: Quantize to Q4_K_M ===")
    final_path = step_quantize(f16_path)

    print(f"\nDone. Load this file in LM Studio:\n  {os.path.abspath(final_path)}")
