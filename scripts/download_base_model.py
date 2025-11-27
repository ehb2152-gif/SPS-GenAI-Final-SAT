# scripts/download_base_model.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import os

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
OUT_DIR = "models/base_llama3_1b"

os.makedirs(OUT_DIR, exist_ok=True)

print("Downloading and saving base model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained(OUT_DIR)

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.save_pretrained(OUT_DIR)

print("✅ Base model saved to:", OUT_DIR)