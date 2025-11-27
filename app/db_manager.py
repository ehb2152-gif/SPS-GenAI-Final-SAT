# app/db_manager.py

import sqlite3
from pathlib import Path

DB_PATH = Path("data/sat_sessions.db")
# Ensure the parent directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# app/db_manager.py

import sqlite3
from pathlib import Path

DB_PATH = Path("data/sat_sessions.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                subject TEXT,
                section TEXT,
                difficulty TEXT,
                prompt TEXT,
                question_text TEXT,
                solutions_included INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS solutions (
                session_id TEXT PRIMARY KEY,
                solution_text TEXT,
                correct_choice TEXT
            )
        """)

def save_session(session_id, subject, section, difficulty, prompt, question_text, include_solutions):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (session_id, subject, section, difficulty, prompt, question_text, int(include_solutions))
        )

def save_solution(session_id, solution_text, correct_choice):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO solutions VALUES (?, ?, ?)",
            (session_id, solution_text, correct_choice)
        )

def get_solution(session_id):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT solution_text, correct_choice FROM solutions WHERE session_id=?",
            (session_id,)
        ).fetchone()

    if not row:
        return None

    return {
        "solution_text": row[0],
        "correct_choice": row[1]
    }

def get_session(session_id):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT subject, section, difficulty, prompt, question_text, solutions_included FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()

    if not row:
        return None

    return {
        "subject": row[0],
        "section": row[1],
        "difficulty": row[2],
        "prompt": row[3],
        "question_text": row[4],
        "solutions_included": bool(row[5]),
    }






'''

-------------------old code --------------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                subject TEXT,
                section TEXT,
                difficulty TEXT,
                prompt TEXT,
                output TEXT,
                solutions_included INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question_num INTEGER,
                user_answer TEXT,
                correct_answer TEXT,
                is_correct INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS correct_answers (
                session_id TEXT,
                question_num INTEGER,
                correct_answer TEXT,
                PRIMARY KEY (session_id, question_num)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS solutions (
                session_id TEXT,
                question_num INTEGER,
                solution_text TEXT,
                PRIMARY KEY (session_id, question_num)
            )
        """)
    print(f"✅ SQLite DB ready at {DB_PATH}")

def save_session(session_id, subject, section, difficulty, prompt, output, include_solutions):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (session_id, subject, section, difficulty, prompt, output, int(include_solutions)),
        )

def get_session(session_id):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT subject, section, difficulty, prompt, output, solutions_included FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "subject": row[0],
        "section": row[1],
        "difficulty": row[2],
        "prompt": row[3],
        "output": row[4],
        "solutions_included": bool(row[5]),
    }

def save_answers(session_id, answers):
    with sqlite3.connect(DB_PATH) as conn:
        for idx, a in enumerate(answers, 1):
            conn.execute(
                "INSERT INTO answers (session_id, question_num, user_answer, correct_answer, is_correct) VALUES (?, ?, ?, ?, ?)",
                (session_id, idx, a["user_answer"], a["correct_answer"], int(a["is_correct"])),
            )

def get_score_summary(session_id):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT COUNT(*), SUM(is_correct) FROM answers WHERE session_id=?",
            (session_id,),
        ).fetchone()
    if not rows or rows[0] == 0:
        return None
    total, correct = rows
    return {"total": total, "correct": correct, "score_percent": round(100 * correct / total, 2)}

# ✅ new helper functions
def save_correct_answers(session_id, answers):
    with sqlite3.connect(DB_PATH) as conn:
        for idx, ans in enumerate(answers, 1):
            conn.execute(
                "INSERT OR REPLACE INTO correct_answers (session_id, question_num, correct_answer) VALUES (?, ?, ?)",
                (session_id, idx, ans),
            )

def get_correct_answers(session_id):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT correct_answer FROM correct_answers WHERE session_id=? ORDER BY question_num",
            (session_id,),
        ).fetchall()
    return [r[0] for r in rows]

def save_solutions(session_id, solutions):
    with sqlite3.connect(DB_PATH) as conn:
        for idx, sol in enumerate(solutions, 1):
            conn.execute(
                "INSERT OR REPLACE INTO solutions (session_id, question_num, solution_text) VALUES (?, ?, ?)",
                (session_id, idx, sol),
            )

def get_solutions(session_id):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT solution_text FROM solutions WHERE session_id=? ORDER BY question_num",
            (session_id,),
        ).fetchall()
    return [r[0] for r in rows]

'''