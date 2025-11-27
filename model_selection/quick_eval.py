# model_selection/quick_eval.py

# 	•	Runs a mini-benchmark:
#	•	Sends prompts for SAT question generation.
#	•	Scores models based on presence of “Step”, “Final Answer”, etc.
#	•	Ranks models by latency & quality.
# Purpose: select the best base model before fine-tuning.


import os, time, json, requests, re
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("Missing HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

# Insert the working models you discovered (from model_probe output)
WORKING = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-2-2b-it"
]

PROMPTS = [
    "Create 1 SAT Algebra I question on linear equations. Show steps. End with: Final Answer: <value>",
    "Create 1 SAT Algebra II question on quadratics (factoring). Show steps. End with: Final Answer: <value>",
    "Create 1 SAT geometry question (area/perimeter). Show steps. End with: Final Answer: <value>",
]

def chat(model, content):
    url = "https://router.huggingface.co/v1/chat/completions"
    payload = {"model": model, "messages":[{"role":"user","content": content}], "max_tokens": 300}
    t0 = time.time()
    r = requests.post(url, headers=HEADERS, json=payload, timeout=90)
    dt = time.time() - t0
    if r.status_code != 200:
        return r.status_code, dt, r.text
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    return 200, dt, text

def score(answer_text):
    s = 0
    if "Step" in answer_text or "step" in answer_text: s += 1
    if "Final Answer:" in answer_text: s += 1
    if re.search(r"= ?\d", answer_text): s += 1
    if re.search(r"\b(x|y)\b", answer_text): s += 1
    return s  # 0..4

rows = []
for m in WORKING:
    total_latency = 0.0
    total_score = 0
    ok = 0
    for p in PROMPTS:
        status, dt, out = chat(m, p)
        total_latency += dt
        if status == 200:
            ok += 1
            total_score += score(out)
    rows.append({
        "model": m,
        "ok/total": f"{ok}/{len(PROMPTS)}",
        "avg_latency_s": round(total_latency/max(1, len(PROMPTS)), 2),
        "rubric_score_0to12": total_score,  # 3 prompts x 4 points
    })

print(json.dumps(sorted(rows, key=lambda r: (-r["rubric_score_0to12"], r["avg_latency_s"])), indent=2))