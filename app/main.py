import os
import torch
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes_generate import router, create_model_and_tokenizer, app_state, MATH_MODEL_DIR, VERBAL_MODEL_DIR, DEVICE

@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_state
    try:
        print("Loading models during startup...")
        app_state["math_tokenizer"], app_state["math_model"] = create_model_and_tokenizer(MATH_MODEL_DIR)
        app_state["verbal_tokenizer"], app_state["verbal_model"] = create_model_and_tokenizer(VERBAL_MODEL_DIR)
        print(f"Models ready on {DEVICE}")
        yield
    except Exception as e:
        print(f"Error loading models: {e}")
        raise RuntimeError("Model loading failed.") from e

app = FastAPI(title="AI SAT Test Generator", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.mount("/", StaticFiles(directory=".", html=True), name="static")