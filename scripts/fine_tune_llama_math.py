# scripts/fine_tune_llama_math.py


"""
Fine-tune Llama-3.2-3B-Instruct on SAT Math (LoRA, PEFT)
- Fixes: labels for loss, batched tokenization, modern BitsAndBytesConfig
- Auto-detects GPU vs CPU
- Quick test generation after training
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model


MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")


# =========================================================
# 0) Setup & config
# =========================================================
load_dotenv()  # loads .env from project root if you run from there
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("Missing HF_TOKEN (set it in your .env or export it before running).")

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../SAT
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sat_instruction_tuning_math.json"
OUT_DIR = PROJECT_ROOT / "models" / "llama3_math_finetuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)



# Hardware detection
has_cuda = torch.cuda.is_available()
has_bf16 = has_cuda and torch.cuda.is_bf16_supported()
device_map = "auto" if has_cuda else None

# Quantization (8-bit if GPU; CPU path uses full precision)
quant_config = None
if has_cuda:
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,           # modern replacement for load_in_8bit arg
        llm_int8_threshold=6.0,
    )

# =========================================================
# 1) Dataset
# =========================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# dataset = load_dataset("json", data_files={"train": str(DATA_PATH)})["train"]
raw_dataset = load_dataset("json", data_files={"train": str(DATA_PATH)})["train"]
split_dataset = raw_dataset.train_test_split(test_size=0.05, seed=42)
train_ds = split_dataset["train"]
eval_ds = split_dataset["test"]

# =========================================================
# 2) Tokenizer & base model
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
# Ensure padding token exists and is consistent
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_kwargs = {
    "token": HF_TOKEN,
}
if has_cuda:
    model_kwargs["device_map"] = device_map
    model_kwargs["quantization_config"] = quant_config

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)

# Align pad token id just in case
if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id

# =========================================================
# 3) Tokenization (batched) + labels for causal LM loss
# =========================================================
def build_prompt(ins, inp, out):
    # Simple SFT format; adjust to your training style as needed
    return (
        f"### Instruction:\n{ins}\n\n"
        f"### Input:\n{inp}\n\n"
        f"### Response:\n{out}"
    )

'''
def build_prompt(ins, inp, out):
    return (
        f"### Instruction:\nSolve the SAT math problem and provide:\n"
        f"- The correct answer choice (A–D)\n"
        f"- A short explanation\n\n"
        f"### Question:\n{inp}\n\n"
        f"### Choices:\n{ins}\n\n"
        f"### Response:\n{out}"
    )
'''

def tokenize(batch):
    prompts = [
        build_prompt(ins, inp, out)
        for ins, inp, out in zip(batch["instruction"], batch["input"], batch["output"])
    ]
    tokens = tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    # Important: provide labels so Trainer can compute loss
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
tokenized_train = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
tokenized_eval  = eval_ds.map(tokenize, batched=True, remove_columns=eval_ds.column_names)

# =========================================================
# 4) LoRA (PEFT)
# =========================================================
peft_config = LoraConfig(
    r=32,       # 16
    lora_alpha=64,     # 32
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],        #["q_proj", "v_proj"]
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # sanity check

# =========================================================
# 5) TrainingArguments
# =========================================================
# Mixed precision flags
use_fp16 = has_cuda and not has_bf16
use_bf16 = has_bf16

training_args = TrainingArguments(
    output_dir=str(OUT_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,       # 4,
    num_train_epochs=3,
    learning_rate=1e-4,           # 2e-4,
    warmup_ratio=0.1,  
    logging_steps=10,
    save_steps=500,           # 200,
    save_total_limit=3,   # 2
    report_to="none",   # no wandb/tensorboard unless you want it
    # fp16=use_fp16,
    # bf16=use_bf16,
    fp16=False,                         # MPS does not support fp16
    bf16=False,                         # MPS does not support bf16
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    # dataloader_pin_memory=has_cuda,  # avoid the Windows warning on CPU
)

# =========================================================
# 6) Trainer & train
# =========================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,   # <-- now evaluation runs!
)
trainer.train()

# =========================================================
# 7) Save LoRA adapter
# =========================================================
model.save_pretrained(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))
print(f"✅ Fine-tuning complete — saved to {OUT_DIR}")

# =========================================================
# 8) Quick sanity check: generation
#     (Uses the in-memory model with LoRA weights attached)
# =========================================================
model.eval()

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device if has_cuda else "cpu")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt = (
    "### Instruction:\nSolve this SAT-style math problem and show the steps.\n\n"
    "### Input:\nIf 3x + 2 = 20, what is x?\n\n"
    "### Response:\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.inference_mode():
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print("\n--- Sample generation ---")
print(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
print("-------------------------")
