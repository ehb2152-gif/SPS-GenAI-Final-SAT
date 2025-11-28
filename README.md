# Domain-Specific AI SAT Tutor
**Group 5:** Eleanor Blum, Kai Pao, Sheethal Heabel

## 1. Project Overview
This project is a domain-specific Generative AI application designed to help students prepare for the SAT. It utilizes two fine-tuned **Llama-3.2-1B** models—one specialized for Math and one for Verbal reasoning—to generate high-quality, unique practice questions, complete with answer choices and detailed explanations.

The system features a **FastAPI** backend for efficient inference and a responsive **HTML/JS** frontend for user interaction. 

## 2. Key Features
* **Single-Pass Inference:** Generates questions, options, and solutions in a single model call for low latency and context consistency.
* **Subject Routing:** Automatically routes requests to the specialized Math or Verbal LoRA adapter based on user input.
* **Robust Parsing:** Uses regex to clean model outputs and ensure structured JSON responses.
* **Persistence:** Saves session data to a local SQLite database (`sat_tutor.db`).
* **Data Pipeline:** Includes custom scripts for extracting and processing SAT content from various sources.

## 3. Repository Structure

app/                    # Main Application Source Code
├── main.py             # FastAPI entry point & static file mounting
├── routes_generate.py  # Core logic: Inference, Cleaning, & DB handling
├── index.html          # Main User Interface (Generator)
├── db_manager.py       # Database utilities
├── prompts.py          # Prompt templates
└── utils_model.py      # Model utility functions

models/                 # Local Fine-Tuned Models (Not in Repo - too large)
├── llama3_math_finetuned/
└── llama3_verbal_finetuned/

data/                   # Processed datasets
data_extraction/        # Scripts for scraping/parsing SAT PDFs and sites
model_selection/        # Scripts used to evaluate base models
core/                   # Core configuration and shared utilities
scripts/                # Training and utility scripts
sat_tutor.db            # SQLite database (Auto-generated)
demo_questions.json     # Storage for saved demo questions
Dockerfile              # Docker container configuration
requirements.txt        # Python dependencies
README.md               # Project documentation

## 4. Setup Instructions (Local)

### Prerequisites
* Python 3.10+
* Git

### Step 1: Clone the Repository
git clone <REPO_LINK_HERE>
cd <REPO_NAME>

### Step 2: Install Dependencies
pip install -r requirements.txt

### Step 3: Download Model Weights
**IMPORTANT:** The fine-tuned model weights are too large for GitHub. You must place them locally:
1.  Ensure the `models/` directory exists in the root.
2.  Place your `llama3_math_finetuned/` and `llama3_verbal_finetuned/` folders inside it.
3.  Ensure you have access to the base `meta-llama/Llama-3.2-1B-Instruct` via Hugging Face.

### Step 4: Run the Application
Run the server from the root directory:

# Set HF_TOKEN if needed for gated models
export HF_TOKEN=your_token_here

python -m uvicorn app.main:app --reload

* **Main Generator:** http://127.0.0.1:8000/
* **API Docs:** http://127.0.0.1:8000/docs

---

## 5. Docker Deployment
The application is containerized for easy deployment.

### Build the Image
docker build -t sat-tutor .

### Run the Container
docker run -p 8000:8000 sat-tutor

---

## 6. Usage Guide
1.  **Generate:** Select a Subject (Math/Verbal) and Difficulty, then click "Generate Question."
2.  **Reveal:** Click "Reveal Solution" to see the answer key and explanation instantly.
3.  **Save:** Click "Save for Demo" to persist high-quality questions to the backend JSON file.
4.  **Playback:** Use the link at the top of the page to visit the Demo Interface and cycle through saved questions.
