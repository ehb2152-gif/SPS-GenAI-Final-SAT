# core/inference.py

#	•	A lightweight text generation client using the Hugging Face Inference API (router endpoint).
#	•	Used for quick testing before fine-tuning.
#	•	HFTextGenerator.generate(prompt) sends a POST to HuggingFace and returns the generated text.
# Purpose: quick question generation without running local model.

import os
import requests
from dotenv import load_dotenv

# Load environment variables (HF_TOKEN, MODEL_ID)
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}


class HFTextGenerator:
    """
    Simple text generation helper using Hugging Face Inference API.
    Used for quick testing or generating SAT-style questions.
    """

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or MODEL_ID
        self.url = "https://router.huggingface.co/v1/chat/completions"

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        """
        Sends a text generation request and returns the model's response.
        """
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

        response = requests.post(self.url, headers=HEADERS, json=payload, timeout=90)

        if response.status_code != 200:
            raise RuntimeError(f"Request failed ({response.status_code}): {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]