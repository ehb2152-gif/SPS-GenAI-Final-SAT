import os
import torch
import time
import re 
import sqlite3 
import uuid 
import json
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import List, Optional

router = APIRouter()

MATH_MODEL_DIR = "models/llama3_math_finetuned"
VERBAL_MODEL_DIR = "models/llama3_verbal_finetuned"
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DB_FILE = "sat_tutor.db"
DEMO_FILE = "demo_questions.json"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, subject TEXT, difficulty TEXT, section TEXT, question TEXT, solution TEXT, timestamp REAL)')
    conn.commit()
    conn.close()

init_db()

class QuestionRequest(BaseModel):
    subject: str
    difficulty: str
    section: str
    include_solution: bool = False 

class QuestionData(BaseModel):
    question: str
    options: List[str] = []
    solution: Optional[str] = None
    subject: str
    difficulty: str
    section: str

class QuestionResponse(BaseModel):
    session_id: str
    question: str
    options: List[str] = []
    solution: Optional[str] = None
    subject: str
    difficulty: str
    section: str
    time_taken_sec: float

DEVICE = "cpu"
app_state = {"math_model": None, "math_tokenizer": None, "verbal_model": None, "verbal_tokenizer": None} 

def create_model_and_tokenizer(model_dir: str):
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, token=hf_token) 
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, device_map=DEVICE, torch_dtype=torch.float32, token=hf_token)
    model = PeftModel.from_pretrained(model, model_dir)
    model = model.merge_and_unload() 
    model.eval() 
    return tokenizer, model

def extract_options(text: str) -> List[str]:
    options = []
    matches = re.findall(r'(?:^|\n)\s*([A-D][\)\.]\s.+)', text)
    for m in matches: options.append(m.strip())
    return options

def clean_model_output(output_text: str) -> (str, str):
    cleaned_output = output_text.replace('**', '').replace('##', '').strip()
    solution_marker_pattern = r'((?:Answer|Correct)(?:\s[A-D]\s)?\s?is correct|Explanation:|Solution:|Your Response:|Answer:|Correct answer is)'
    question_content = cleaned_output
    solution_content = None
    match = re.search(solution_marker_pattern, cleaned_output, re.DOTALL | re.IGNORECASE)
    if match:
        split_index = match.start()
        question_segment = cleaned_output[:split_index].strip()
        solution_content = cleaned_output[split_index:].strip()
        preamble_markers = r'^(?:Choose the correct answer.*?|Please select.*?|The question should include.*?|Question:|Here is your question:|Here is your SAT math question:|Please answer the question below\.|Here\'s the question:|Your response should be presented in the format:)'
        current_content = question_segment
        while re.match(preamble_markers, current_content, re.IGNORECASE):
            current_content = re.sub(preamble_markers, '', current_content, flags=re.IGNORECASE).strip()
        question_content = current_content.strip()
    else:
        question_content = cleaned_output
    return question_content, solution_content

def generate_text_from_model(tokenizer, model, prompt: str):
    start_time = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    norm_prompt = re.sub(r'\s+', ' ', prompt).strip()
    norm_gen_text = re.sub(r'\s+', ' ', generated_text).strip()
    if norm_gen_text.startswith(norm_prompt): clean_output = generated_text[len(prompt):].strip()
    else: clean_output = generated_text.replace(prompt, "").strip()
    return clean_output, time.time() - start_time

def save_session_to_sqlite(session_id, subject, difficulty, section, question, solution):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)", (session_id, subject, difficulty, section, question, solution, time.time()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving to SQLite: {e}")

@router.post("/generate", response_model=QuestionResponse)
async def generate_question_endpoint(request: QuestionRequest):
    if app_state["math_model"] is None: raise HTTPException(status_code=503, detail="Models not loaded.")
    if request.subject.lower() == "math": tokenizer = app_state["math_tokenizer"]; model = app_state["math_model"]
    elif request.subject.lower() == "verbal": tokenizer = app_state["verbal_tokenizer"]; model = app_state["verbal_model"]
    else: raise HTTPException(status_code=400, detail="Invalid subject.")
    instruction_prompt = f"You are an expert SAT test generator. Create a unique {request.difficulty} difficulty {request.subject} question focused on {request.section}. The question should be multiple choice. Provide the question, options, correct answer, and explanation."
    print(f"Generating question for {request.subject}...")
    full_output, time_taken = generate_text_from_model(tokenizer, model, instruction_prompt)
    question_text, solution_text = clean_model_output(full_output)
    options_list = extract_options(question_text)
    session_id = str(uuid.uuid4())
    safe_solution = solution_text if solution_text else ""
    save_session_to_sqlite(session_id, request.subject, request.difficulty, request.section, question_text, safe_solution)
    return QuestionResponse(session_id=session_id, question=question_text, options=options_list, solution=solution_text, subject=request.subject, difficulty=request.difficulty, section=request.section, time_taken_sec=round(time_taken, 2))

@router.post("/save", status_code=status.HTTP_201_CREATED)
async def save_question_for_demo(data: QuestionData):
    try:
        if os.path.exists(DEMO_FILE):
            with open(DEMO_FILE, 'r') as f: questions = json.load(f)
        else: questions = []
        questions.append(data.dict())
        with open(DEMO_FILE, 'w') as f: json.dump(questions, f, indent=4)
        return {"message": "Saved!", "count": len(questions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/demo/questions", response_model=List[QuestionData])
async def get_demo_questions():
    try:
        if os.path.exists(DEMO_FILE):
            with open(DEMO_FILE, 'r') as f: return json.load(f)
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))