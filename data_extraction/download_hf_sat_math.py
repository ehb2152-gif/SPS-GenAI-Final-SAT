# data_extraction/download_hf_sat_math.py

#	•	Downloads a public dataset (RyanYr/reflect_agieval-sat-test_t0) from HuggingFace Datasets Hub.
#	•	Converts it into a local .json file for later processing.
# Purpose: get high-quality SAT Math problems.

from datasets import load_dataset
import json
from pathlib import Path

# === Output path ===
OUT_PATH = Path("data/processed/hf_math_sat.json")

def download_reference_dataset():
    """
    Downloads the SAT portion of AGIEval from Hugging Face
    and saves it locally as data/processed/hf_math_sat.json
    """
    print("📥 Downloading RyanYr/reflect_agieval-sat-test_t0 ...")
    ds = load_dataset("RyanYr/reflect_agieval-sat-test_t0")

    # Convert Hugging Face dataset to list of dicts
    data = [dict(item) for item in ds["train"]]

    # Ensure output directory exists
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save locally
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved reference dataset to {OUT_PATH}")
    print(f"📊 Total examples: {len(data)}")

if __name__ == "__main__":
    download_reference_dataset()
