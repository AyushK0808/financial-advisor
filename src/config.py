# src/config.py - Simple Configuration

import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """Simple configuration"""
    
    # MongoDB
    mongodb_url: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    mongodb_db: str = os.getenv("MONGODB_DB", "financial_advisor")
    
    # ChromaDB
    chromadb_path: str = os.getenv("CHROMADB_PATH", "./vector_store/chromadb")
    chromadb_host: str = os.getenv("CHROMADB_HOST", "localhost")
    chromadb_port: int = int(os.getenv("CHROMADB_PORT", "8000"))
    
    # LLM Model
    model_name: str = os.getenv("MODEL_NAME", "microsoft/DialoGPT-small")
    model_path: str = os.getenv("MODEL_PATH", "./models/financial_model")
    fine_tune_epochs: int = int(os.getenv("FINE_TUNE_EPOCHS", "3"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "5e-5"))
    max_length: int = int(os.getenv("MAX_LENGTH", "256"))
    
    # SearXNG
    searxng_url: str = os.getenv("SEARXNG_URL", "http://localhost:8888")
    
    # Settings
    yahoo_finance_enabled: bool = os.getenv("YAHOO_FINANCE_ENABLED", "true").lower() == "true"
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    save_user_data: bool = os.getenv("SAVE_USER_DATA", "true").lower() == "true"
    max_portfolio_size: int = int(os.getenv("MAX_PORTFOLIO_SIZE", "50"))

config = Config()