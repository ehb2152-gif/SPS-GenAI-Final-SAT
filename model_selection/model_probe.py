# model_selection/model_probe.py

#	•	Tests multiple Hugging Face models by making both chat and text-generation API calls.
#	•	Records status_code, latency, and short output samples.
# Purpose: discover which models work with your HF token (some models require approval).

import os, time, json, requests
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("Missing HF_TOKEN")

CANDIDATES = [
    "HuggingFaceH4/zephyr-7b-beta",
    "microsoft/Phi-3-mini-4k-instruct",
    "tiiuae/falcon-7b-instruct",
    "Qwen/Qwen2.5-Math-1.5B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

def try_chat(model):
    url = "https://router.huggingface.co/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi!"}],
        "max_tokens": 16,
    }
    t0 = time.time()
    r = requests.post(url, headers=HEADERS, json=payload, timeout=60)
    dt = time.time() - t0
    return r.status_code, dt, r.text[:300]

def try_text(model):
    url = "https://router.huggingface.co/v1/text-generation"
    payload = {
        "model": model,
        "inputs": "Say hi in one short sentence.",
        "parameters": {"max_new_tokens": 16},
    }
    t0 = time.time()
    r = requests.post(url, headers=HEADERS, json=payload, timeout=60)
    dt = time.time() - t0
    return r.status_code, dt, r.text[:300]

rows = []
for m in CANDIDATES:
    chat = try_chat(m)
    text = try_text(m)
    rows.append({
        "model": m,
        "chat_status": chat[0], "chat_latency_s": round(chat[1],2), "chat_sample": chat[2],
        "text_status": text[0], "text_latency_s": round(text[1],2), "text_sample": text[2],
    })

print(json.dumps(rows, indent=2))