# scripts/fine_tune_llama_verbal.py

"""
Fine-tune Llama-3.2-3B-Instruct on SAT Verbal (LoRA, PEFT)
- Compatible with dataset prepared by prepare_verbal_dataset.py
- Auto-detects GPU/CPU
- Saves trained LoRA weights under models/llama3_verbal_finetuned
"""

import os
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
# 0) Setup & Config
# =========================================================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("❌ Missing HF_TOKEN. Please set it in your .env file.")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sat_instruction_tuning_verbal.json"
OUT_DIR = PROJECT_ROOT / "models" / "llama3_verbal_finetuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)



# Hardware detection
has_cuda = torch.cuda.is_available()
has_bf16 = has_cuda and torch.cuda.is_bf16_supported()
device_map = "auto" if has_cuda else None

quant_config = None
if has_cuda:
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

# =========================================================
# 1) Dataset
# =========================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"❌ Dataset not found: {DATA_PATH}")

# dataset = load_dataset("json", data_files={"train": str(DATA_PATH)})["train"]
raw_dataset = load_dataset("json", data_files={"train": str(DATA_PATH)})["train"]
split_dataset = raw_dataset.train_test_split(test_size=0.05, seed=42)
train_ds = split_dataset["train"]
eval_ds = split_dataset["test"]

# =========================================================
# 2) Tokenizer & Model
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_kwargs = {"token": HF_TOKEN}
if has_cuda:
    model_kwargs["device_map"] = device_map
    model_kwargs["quantization_config"] = quant_config

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)

if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id

# =========================================================
# 3) Tokenization
# =========================================================
def build_prompt(instruction, context, answer):
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Passage:\n{context}\n\n"
        f"### Response:\n{answer}"
    )

def tokenize(batch):
    prompts = [
        build_prompt(ins, inp, out)
        for ins, inp, out in zip(batch["instruction"], batch["input"], batch["output"])
    ]
    tokens = tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_train = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
tokenized_eval = eval_ds.map(tokenize, batched=True, remove_columns=eval_ds.column_names)

# =========================================================
# 4) LoRA Configuration
# =========================================================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.config.use_cache = False

# =========================================================
# 5) Training Arguments
# =========================================================
use_fp16 = has_cuda and not has_bf16
use_bf16 = has_bf16

training_args = TrainingArguments(
    output_dir=str(OUT_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,     # 4,
    num_train_epochs=2,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    fp16=use_fp16,
    bf16=use_bf16,
    dataloader_pin_memory=has_cuda,
)

# =========================================================
# 6) Trainer
# =========================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)
trainer.train()

# =========================================================
# 7) Save Fine-tuned Model
# =========================================================
model.save_pretrained(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))
print(f"✅ Verbal fine-tuning complete — saved to {OUT_DIR}")

# =========================================================
# 8) Quick Sanity Check
# =========================================================
model.eval()
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
model.to(device)
prompt = (
    "### Instruction:\nRead the passage and answer the question concisely.\n\n"
    "### Passage:\nIn the passage, the author suggests that scientific progress often creates new moral dilemmas.\n"
    "### Question:\nWhat is the author’s main concern?\n\n### Response:\n"
)

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device if has_cuda else "cpu")
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
