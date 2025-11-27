# data_extraction/prepare_math_dataset.py

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

def format_sat(example):
    instruction = "Solve the following SAT math problem and explain your reasoning."
    input_text = example["problem"]
    response = example.get("solution", "")
    return {
        "instruction": instruction,
        "input": input_text,
        "output": response
    }

# Load your dataset
dataset = load_dataset("json", data_files={"train": "data/processed/hf_math_sat.json"})

# Apply formatting
formatted = dataset["train"].map(format_sat)

# Save processed file
formatted.to_json("data/processed/sat_instruction_tuning_math.json", orient="records", lines=True)
print("✅ Saved processed dataset to data/processed/sat_instruction_tuning_math.json")
