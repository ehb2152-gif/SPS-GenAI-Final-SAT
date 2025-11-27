# app/test_inference_local.py

"""
Quick sanity check for fine-tuned Llama-3.2-1B-Instruct model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Choose which model to test
MODEL_PATH = "models/llama3_math_finetuned"   # or "models/llama3_verbal_finetuned"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model.to(device)
model.eval()

prompt = (
    "### Instruction:\nSolve this SAT-style math problem and show the steps.\n\n"
    "### Input:\nIf 4x + 8 = 20, what is x?\n\n"
    "### Response:\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
    gen = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

print(tokenizer.decode(gen[0], skip_special_tokens=True))
