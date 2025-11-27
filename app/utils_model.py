# app/utils_model.py

import gc
import os
import time
import random
import uuid
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from app.db_manager import (
    init_db,
    save_session,
    save_solution,
    get_session, 
)
from app.prompts import (
    question_prompt,
    solution_prompt,
    reveal_prompt
)

load_dotenv()
init_db()
# torch.set_default_dtype(torch.float16)

class DualModelManager:

    def __init__(self):
        """Load math + verbal fine-tuned models."""

        # ---------------- Select device -----------------
        '''
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        '''
        self.device = "cpu"

        # -------------------------------------------------------
        # Load 2 models (Math & Verbal) & tokenizer for both 
        # -------------------------------------------------------
        math_path = os.getenv("MODEL_PATH_MATH", "models/llama3_math_finetuned")
        verbal_path = os.getenv("MODEL_PATH_VERBAL", "models/llama3_verbal_finetuned")

        print(f"🧠 Loading Math model from {math_path}")
        self.tokenizer_math = AutoTokenizer.from_pretrained(math_path)
        self.model_math = AutoModelForCausalLM.from_pretrained(math_path,low_cpu_mem_usage=True).to(self.device)

        print(f"📚 Loading Verbal model from {verbal_path}")
        self.tokenizer_verbal = AutoTokenizer.from_pretrained(verbal_path)
        self.model_verbal = AutoModelForCausalLM.from_pretrained(verbal_path).to(self.device)

        print(f"✅ Both models ready on {self.device}")

    # -------------------------------------------------------
    # MODEL SELECTOR
    # -------------------------------------------------------
    def choose_model(self, prompt: str, subject: str | None = None) -> str:

        # --------------------------------
        # 1. SUBJECT PARAM OVERRIDES ALL
        # --------------------------------
        if subject:
            s = subject.lower()
            if "math" in s:
                return "math"
            if "verbal" in s or "reading" in s or "english" in s:
                return "verbal"

        # --------------------------------
        # 2. KEYWORD-BASED DETECTION
        # --------------------------------

        math_keywords = [
            "solve", "math", "algebra", "geometry", "equation",
            "trigonometry", "linear equation", "function", "graph",
            "inequality", "polynomial", "variable", "integer", "ratio",
            "percent", "probability"
        ]

        verbal_keywords = [
            "passage", "reading", "context", "author", "tone",
            "inference", "meaning", "vocabulary", "synonym",
            "main idea", "evidence", "sentence", "grammar",
            "interpret", "theme"
        ]

        prompt_lower = prompt.lower()

        # math detection
        if any(k in prompt_lower for k in math_keywords):
            return "math"

        # verbal detection
        if any(k in prompt_lower for k in verbal_keywords):
            return "verbal"

        # --------------------------------
        # 3. FALLBACK
        # --------------------------------
        return "verbal"

    # -------------------------------------------------------
    # GENERATE ONE QUESTION + BACKEND SOLUTION
    # -------------------------------------------------------
    def generate(self, prompt: str, subject: str, difficulty: str, include_solutions: bool = False):

        # ---------------- MODEL SELECT ----------------
        model_type = self.choose_model(prompt, subject)          # Model selector function
        tokenizer = self.tokenizer_math if model_type == "math" else self.tokenizer_verbal
        model = self.model_math if model_type == "math" else self.model_verbal

        # ---------------- Generate the QUESTION PROMPT ----------------
        full_prompt = question_prompt(
            subject=subject,
            topic=prompt,
            difficulty=difficulty
        )
        
        inputs = tokenizer(full_prompt, return_tensors="pt",truncation=True, max_length=256).to(self.device)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        # ---------------- RANDOMNESS SEED ----------------
        # Seed every request with a fresh high-entropy value
        seed = time.time_ns() % (2**32)
        torch.manual_seed(seed)
        random.seed(seed)

        # ===== GENERATE QUESTION =====
        gc.collect()
        # if torch.backends.mps.is_available(): 
        #     torch.mps.empty_cache()
        with torch.inference_mode():
            '''
            q_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=pad_id,
            )
            '''
            q_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,       # ← change
                temperature=0.7,      # ← use your ENV vars
                top_p=0.8,
                pad_token_id=pad_id,
                repetition_penalty=1.15,
            )

        raw_question = tokenizer.decode(q_ids[0], skip_special_tokens=True)

        # Remove repeated prompt text
        if "Question:" in raw_question:
            # question_text = raw_question.split("Question:", 1)[-1].strip()
            question_text = raw_question.replace(full_prompt, "").strip()
        else:
            question_text = raw_question.strip()

        # ---------------- CREATE SESSION ----------------
        session_id = str(uuid.uuid4())[:8]

        save_session(
            session_id=session_id,
            subject=subject,
            section=prompt,
            difficulty=difficulty,
            prompt=prompt,
            question_text=question_text,
            include_solutions=include_solutions,
        )

        # ---------------- SOLUTION GENERATION ----------------
        solution_text_prompt = solution_prompt(question_text)

        inputs2 = tokenizer(solution_text_prompt, return_tensors="pt",truncation=True,max_length=256).to(self.device)

        # ---------------- RANDOMNESS SEED ----------------
        # Seed every request with a fresh high-entropy value
        seed = time.time_ns() % (2**32)
        torch.manual_seed(seed)
        random.seed(seed)

        # ===== GENERATE SOLUTION =====
        gc.collect()
        # if torch.backends.mps.is_available(): 
        #    torch.mps.empty_cache()
        with torch.inference_mode():
            '''
            sol_ids = model.generate(
                **inputs2,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=pad_id,
            )
            '''
            sol_ids = model.generate(
                **inputs2,
                max_new_tokens=64,
                do_sample=False,       # ← change
                # temperature=0.65,      # ← use your ENV vars
                # top_p=0.6,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # raw_solution = tokenizer.decode(sol_ids[0], skip_special_tokens=True).strip()
        raw_solution = tokenizer.decode(sol_ids[0], skip_special_tokens=True)
        # raw_solution = raw_solution.replace(solution_text_prompt, "").strip()
        
        # Remove any echo of the "### Instruction:" or "### Question:" blocks
        lines = raw_solution.split("\n")
        cleaned_lines = []

        for line in lines:
            # filter out prompt-echo lines
            if "Instruction:" in line: continue
            if "Question:" in line: continue
            if "Response:" in line: continue
            if "Example:" in line: continue
            if "First line" in line: continue
            if "Second line" in line: continue

            cleaned_lines.append(line.strip())

        raw_solution = "\n".join([l for l in cleaned_lines if l])   

        # ---------------- Extract the correct choice ----------------
        first_line = raw_solution.split("\n")[0].strip()

        import re
        # match = re.match(r"^[A-D]$", first_line)
        match = re.search(r"\b([A-D])\b", raw_solution)
        correct_choice = match.group(0) if match else None

        # ---------------- Extract explanation ----------------
        explanation = "\n".join(raw_solution.split("\n")[1:]).strip()

        # Save solution to DB
        save_solution(session_id, explanation, correct_choice)

        # ---------------- RETURN RESPONSE ----------------
        if include_solutions:
            return {
                "session_id": session_id,
                "model_used": model_type,
                "difficulty": difficulty,
                "question": question_text,
                "correct_choice": correct_choice,
                "solution": explanation,
            }

        return {
            "session_id": session_id,
            "model_used": model_type,
            "difficulty": difficulty,
            "question": question_text,
        }

    # -------------------------------------------------------
    # REVEAL SOLUTIONS
    # -------------------------------------------------------
    def reveal_solutions(self, session_id: str):
        """Generate only A–D answers."""
        session = get_session(session_id)
        if not session:
            return {"error": "Invalid session_id"}

        subject = session["subject"]

        reveal_prompt = (
            "### Instruction:\n"
            "Return ONLY the correct answer letter (A–D) for each question.\n\n"
            "### Questions:\n"
            f"{session['output']}\n\n"
            "### Response:\n"
        )

        tokenizer = self.tokenizer_math if subject == "math" else self.tokenizer_verbal
        model = self.model_math if subject == "math" else self.model_verbal

        inputs = tokenizer(reveal_prompt, return_tensors="pt").to(self.device)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        with torch.inference_mode():
            ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=pad_id,
            )

        out = tokenizer.decode(ids[0], skip_special_tokens=True)
        return {"session_id": session_id, "solutions": out}






'''

    # ---------------- old code ----------------
    # -------------------------------------------------------
    # SIMPLE GENERATE() — ONLY TAKES USER PROMPT
    # -------------------------------------------------------
    def generate(
        self,
        prompt: str,
        subject: str,
        difficulty: str,
        include_solutions: bool,
        num_questions: int,
        max_new_tokens: int = 350,
    ):
        """Generate SAT questions from a single flexible prompt, 
        AND secretly store full solutions in DB."""

        model_type = self.choose_model(prompt, subject)
        tokenizer = self.tokenizer_math if model_type == "math" else self.tokenizer_verbal
        model = self.model_math if model_type == "math" else self.model_verbal

        # Build user prompt
        sys_prompt = f"Create SAT {model_type} problems of {difficulty} difficulty."

        if include_solutions:
            user_prompt = "Include full step-by-step solutions."
        else:
            user_prompt = "Do NOT include solutions."

        full_prompt = (
            f"### Instruction:\n{sys_prompt}\n{user_prompt}\n\n"
            f"### Input:\n{prompt}\n\n"
            f"### Response:\n"
        )

        # --------------------- Generate questions ---------------------
        inputs = tokenizer(full_prompt, return_tensors="pt").to(self.device)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        with torch.inference_mode():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )

        output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # Save session
        session_id = str(uuid.uuid4())[:8]
        save_session(
            session_id,
            model_type,
            subject,
            difficulty,
            full_prompt,
            output,
            include_solutions,
        )

        # ---------------- SECRET BACKEND SOLUTION GENERATION -----------------
        from app.db_manager import save_solutions
        import re
        

        solution_prompt = (
            f"### Instruction:\n"
            f"Provide extremely detailed step-by-step solutions for EACH question.\n"
            f"Number solutions like: 1), 2), 3)...\n\n"
            f"### Questions:\n{output}\n\n"
            f"### Response:\n"
        )


        solution_prompt = (
            f"### Instruction:\n"
            f"You are an SAT math solution generator.\n"
            f"Provide ONLY the detailed solutions.\n"
            f"Do NOT repeat the questions.\n"
            f"Do NOT include ### Response blocks.\n\n"
            f"STRICT FORMAT:\n"
            f"1) Detailed solution for question 1:\n"
            f"2) Detailed solution for question 2:\n"
            f"3) Detailed solution for question 3:\n\n"
            f"Write full math reasoning under each number.\n"
            f"Do NOT leave anything blank.\n"
            f"Do NOT output placeholders.\n"
            f"Fill in the real solutions.\n\n"
            f"### Questions:\n{output}\n\n"
            f"### Response:\n"
        )

        inputs2 = tokenizer(solution_prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            sol_ids = model.generate(
                **inputs2,
                max_new_tokens=1000,
                do_sample=False,
                pad_token_id=pad_id,
            )

        solutions_raw = tokenizer.decode(sol_ids[0], skip_special_tokens=True)

        # Extract continuation
        if "### Response:" in solutions_raw:
            sol_text = solutions_raw.rsplit("### Response:", 1)[1].strip()
        else:
            sol_text = solutions_raw

        # Split solutions by question number
        solutions_clean = []
        for i in range(1, num_questions + 1):
            # pattern = rf"{i}\)\s*(.*?)(?=\n\d\)|$)"
            # pattern = rf"{i}[\.\)]\s*(.*?)(?=\n{i+1}[\.\)]|$)"
            pattern = rf"{i}[\.\)]\s*(.*?)(?=\n{i+1}[\.\)]|\Z)"
            m = re.search(pattern, sol_text, re.DOTALL)
            if m:
                solutions_clean.append(m.group(1).strip())
            else:
                solutions_clean.append("Solution not extracted.")

        save_solutions(session_id, solutions_clean)

        # ---------------- RETURN ----------------
        return {
            "session_id": session_id,
            "model_used": model_type,
            "difficulty": difficulty,
            "raw_output": output,
        }

'''


