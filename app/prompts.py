# app/prompts.py

"""
Centralized prompt templates for SAT Tutor generation.
This keeps prompts clean, editable, and consistent.
"""

# -------------------------------------------
# QUESTION GENERATION PROMPT
# -------------------------------------------
def question_prompt(subject: str, topic: str, difficulty: str) -> str:
    return f"""
Write one SAT-style {subject.lower()} question about {topic}.
Include exactly four answer choices labeled:
A)
B)
C)
D)
Do NOT reveal the correct answer.
Do NOT include explanations or meta instructions.

Question:
""".strip()
     


# -------------------------------------------
# SOLUTION GENERATION PROMPT
# -------------------------------------------
def solution_prompt(question_text: str) -> str:
    return f"""
### Instruction:
Provide ONLY the correct answer choice (A, B, C, or D) on the first line.
Provide a short explanation on the second line.

### Question:
{question_text}

### Response:
""".strip()

# -------------------------------------------
# OPTIONAL – REVEAL PROMPT (if you need model-generated answer-only)
# -------------------------------------------
def reveal_prompt(question_text: str) -> str:
    return f"""
Return ONLY the correct answer letter (A, B, C, or D) for the question below.

Question:
{question_text}

Answer:
""".strip()










 # ---------------- Generate the QUESTION PROMPT ----------------

'''

        Write one SAT-style {subject.lower()} question about {topic}.
        Include exactly four answer choices labeled:
        A)
        B)
        C)
        D)
        Do NOT reveal the correct answer.
        Do NOT include explanations or meta instructions.

        Question:
        """.strip()

        

    return f"""
            Write one SAT-style {subject.lower()} question about {topic}.
            Include exactly four answer choices labeled:
            A)
            B)
            C)
            D)
            Do NOT reveal the correct answer.
            Do NOT include explanations or meta instructions.

            Question:
            """.strip()



        full_prompt = f"""
        Create ONE SAT-style {subject.lower()} question.

        Topic: {prompt}
        Difficulty: {difficulty}

        Include answer choices labeled:
        A)
        B)
        C)
        D)

        Do NOT reveal the correct answer.
        Do NOT include boxed answers or meta instructions.

        Question:
        """.strip()

        
        
        
        full_prompt = f"""
        Write one SAT-style {subject.lower()} question about {prompt}.
        Include four answer choices:
        A)
        B)
        C)
        D)
        Do not give the correct answer.
        """.strip()
        
        

        full_prompt = f"""
        Write one SAT-style {subject.lower()} question about {prompt}.
        Give four answer choices labeled A), B), C), and D).
        Do NOT give the correct answer.

        Question:
        """.strip()

        
        full_prompt = """
        Write one SAT-style math question.
        Give A), B), C), D).
        Do NOT give the answer.
        Question:
        """
'''

# ---------------- SOLUTION GENERATION ----------------
'''
    return f"""
Solve the SAT question below.

First line MUST contain ONLY the correct answer letter (A, B, C, or D).
Second line MUST contain a short explanation.

Example:
A
Because ...

Question:
{question_text}
""".strip()



    return f"""
            Solve the SAT question below.

            First line MUST contain ONLY the correct answer letter (A, B, C, or D).
            Second line MUST contain a short explanation.

            Example:
            A
            Because ...

            Question:
            {question_text}
            """.strip()



        solution_prompt = f"""
        Solve the SAT question below.

        First line MUST contain ONLY the correct answer letter (A, B, C, or D).
        Second line: give a short explanation.

        Example:
        A
        Because ...

        Question:
        {question_text}
        """.strip()
        

        solution_prompt = f"""
        Solve the SAT question below.

        First line: ONLY the correct answer letter (A, B, C, or D).
        Second line: A short explanation.

        Question:
        {question_text}
        """.strip()
        

        solution_prompt = (
            f"### Instruction:\n"
            f"Provide extremely detailed step-by-step solutions for EACH question.\n"
            f"Number solutions like: 1), 2), 3)...\n\n"
            f"### Questions:\n{output}\n\n"
            f"### Response:\n"
        )
'''