import os
import torch
import time
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Router Setup 
router = APIRouter()

# Configuration Constants 
MATH_MODEL_DIR = "models/llama3_math_finetuned"
VERBAL_MODEL_DIR = "models/llama3_verbal_finetuned"
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Request/Response Schemas 
class QuestionRequest(BaseModel):
    subject: str # "Math" or "Verbal"
    difficulty: str # "Easy", "Medium", "Hard"
    section: str # "algebra", "vocabulary", etc.
    include_solution: bool = False

class QuestionResponse(BaseModel):
    question: str
    solution: str | None = None
    time_taken_sec: float

# Model State and Loading 
DEVICE = "cpu"
app_state = {"math_model": None, "math_tokenizer": None, "verbal_model": None, "verbal_tokenizer": None}
MODELS_LOADED = False

def create_model_and_tokenizer(model_dir: str):
    """Loads the base model and then loads the PEFT fine-tuning adapters."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}. Did fine-tuning finish successfully?")

    print(f"Loading tokenizer for {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, token=hf_token)

    print(f"Loading base model {BASE_MODEL_ID}...")
    
    # Load model onto CPU using standard precision (float32)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map=DEVICE, 
        torch_dtype=torch.float32, 
        token=hf_token
    )

    # Load Fine-Tuned Adapter (PEFT)
    print(f"Loading PEFT adapter from {model_dir}...")
    model = PeftModel.from_pretrained(model, model_dir)
    model = model.merge_and_unload() 
    model.eval() 
    
    print(f"✅ Model loaded: {model_dir}")
    return tokenizer, model

def load_models_once():
    """Lazy loads models when the first API call is made."""
    global MODELS_LOADED
    if not MODELS_LOADED:
        print(f"Setting device to: {DEVICE}")
        print("--- STARTING MODEL INITIALIZATION ---")
        app_state["math_tokenizer"], app_state["math_model"] = create_model_and_tokenizer(MATH_MODEL_DIR)
        app_state["verbal_tokenizer"], app_state["verbal_model"] = create_model_and_tokenizer(VERBAL_MODEL_DIR)
        print("--- MODEL INITIALIZATION COMPLETE ---")
        MODELS_LOADED = True


def generate_text_from_model(tokenizer, model, prompt: str):
    """Utility function to run inference on the model."""
    start_time = time.time()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clean_output = generated_text.replace(prompt, "", 1).strip()
    
    time_taken = time.time() - start_time
    
    return clean_output, time_taken

# API Endpoints

@router.post("/generate", response_model=QuestionResponse)
async def generate_question_endpoint(request: QuestionRequest):
    """Endpoint that the website calls to generate a question."""
    
    # Ensure models are loaded before processing the request
    try:
        load_models_once()
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


    subject = request.subject
    section = request.section
    difficulty = request.difficulty
    
    if subject == "Math":
        tokenizer = app_state["math_tokenizer"]
        model = app_state["math_model"]
        model_name = "Math"
    elif subject == "Verbal":
        tokenizer = app_state["verbal_tokenizer"]
        model = app_state["verbal_model"]
        model_name = "Verbal"
    else:
        raise HTTPException(status_code=400, detail="Invalid subject provided.")

    instruction_prompt = (
        f"You are an expert SAT test generator. Create a unique {difficulty} difficulty "
        f"{subject} question focused on {section}. The question should be multiple choice."
    )
    
    print(f"Generating question for {subject}/{difficulty}/{section} using {model_name} model...")
    
    full_output, time_taken = generate_text_from_model(tokenizer, model, instruction_prompt)

    # Post-Processing to Separate Question and Solution 
    question_text = full_output
    solution_text = None
    
    split_keywords = ["Solution:", "Answer:", "Explanation:"]
    for keyword in split_keywords:
        if keyword in full_output:
            parts = full_output.split(keyword, 1)
            question_text = parts[0].strip()
            solution_text = keyword + parts[1].strip()
            break
            
    if not request.include_solution:
        solution_text = None
    
    if not question_text:
        question_text = f"Failed to generate output from the model. Prompt was: {instruction_prompt}"
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=question_text)

    return QuestionResponse(
        question=question_text,
        solution=solution_text,
        time_taken_sec=round(time_taken, 2)
    )