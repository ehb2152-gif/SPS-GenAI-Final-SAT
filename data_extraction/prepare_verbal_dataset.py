# data_extraction/prepare_verbal_dataset.py

# 	•	Load the processed JSON and convert each entry into instruction-tuning format:
'''
        {
        "instruction": "Solve the following SAT math problem...",
        "input": "problem statement here",
        "output": "solution or explanation here"
        }
'''
# Purpose: prepare the dataset for supervised fine-tuning (SFT) with transformers.Trainer.




from datasets import load_dataset
import json

def format_sat_verbal(example):
    """
    Converts each verbal reasoning question into an instruction-tuning triple.
    """
    instruction = "Read the two texts and answer the question based on them."
    
    # Combine the passage text into one input block
    input_text = example["question"]
    
    # Create a structured answer explanation (use rationale if available)
    answer = example.get("answer", "")
    rationale = example.get("rationale", "")
    if rationale:
        response = f"The correct answer is {answer}. Explanation: {rationale}"
    else:
        response = f"The correct answer is {answer}."
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": response
    }

# Load the raw JSON
dataset = load_dataset("json", data_files={"train": "data/processed/sat_questionbank_verbal.json"})

# Apply transformation
formatted = dataset["train"].map(format_sat_verbal)

# Save processed dataset for fine-tuning
output_path = "data/processed/sat_instruction_tuning_verbal.json"
formatted.to_json(output_path, orient="records", lines=True)

print(f"✅ Saved processed verbal dataset to {output_path}")
