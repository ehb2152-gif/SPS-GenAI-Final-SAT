# data_extraction/extract_math.py

#	•	Use pdfplumber to extract question text, choices, and correct answers from SAT PDFs.
#	•	Regex cleans and structures questions into JSON.
# Purpose: convert raw PDFs → structured JSON (id, question, choices, answer, rationale, etc.).

import re, json
from pathlib import Path
import pdfplumber

# === Paths ===
RAW_PATH = Path("data/raw/SAT Suite Question Bank - Math.pdf")
OUT_PATH = Path("data/processed/sat_questionbank_math.json")

def extract_sat_math_questions(pdf_path):
    print(f"📘 Extracting questions from {pdf_path.name} ...")

    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    # Split into chunks starting with "Question ID"
    chunks = re.split(r"\bQuestion ID\b", text)[1:]  # skip header
    questions = []

    for chunk in chunks:
        # Basic metadata
        qid = re.search(r"ID:\s*([A-Za-z0-9]+)", chunk)
        question_text = re.search(r"ID:\s*[A-Za-z0-9]+\s*(.*?)\nA\.", chunk, re.S)
        choices = re.findall(r"([A-D])\.\s*(.*?)(?=\n[A-D]\.|ID:|Correct Answer:|$)", chunk, re.S)
        correct = re.search(r"Correct Answer:\s*([A-D])", chunk)
        rationale = re.search(r"Rationale(.*?)(?=Question Difficulty:|$)", chunk, re.S)
        difficulty = re.search(r"Question Difficulty:\s*(\w+)", chunk)
        domain = re.search(r"Domain:\s*(.*?)\n", chunk)
        skill = re.search(r"Skill:\s*(.*?)\n", chunk)

        if not (qid and question_text and choices):
            continue

        questions.append({
            "id": qid.group(1).strip(),
            "question": " ".join(question_text.group(1).split()),
            "choices": [f"{c[0]}. {c[1].strip()}" for c in choices],
            "answer": correct.group(1).strip() if correct else None,
            "rationale": rationale.group(1).strip() if rationale else None,
            "difficulty": difficulty.group(1).strip() if difficulty else None,
            "domain": domain.group(1).strip() if domain else None,
            "skill": skill.group(1).strip() if skill else None
        })

    print(f"✅ Extracted {len(questions)} math questions.")
    return questions

if __name__ == "__main__":
    qs = extract_sat_math_questions(RAW_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(qs, open(OUT_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"💾 Saved → {OUT_PATH}")
