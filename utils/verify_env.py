# utils/verify_env.py
import os, torch, sys
from dotenv import load_dotenv

load_dotenv()

print("🔍 Environment Verification")
print("-" * 40)
print(f"Python: {sys.version}")
print(f"Path: {sys.executable}")
print(f"Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
print(f"HF_TOKEN present: {bool(os.getenv('HF_TOKEN'))}")
print(f"MODEL_ID: {os.getenv('MODEL_ID', '❌ not set')}")
print("-" * 40)