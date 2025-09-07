# src/config.py - Enhanced Configuration with Synthesis Settings

import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """Enhanced configuration with synthesis and generalization settings"""

    # MongoDB
    mongodb_url: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    mongodb_db: str = os.getenv("MONGODB_DB", "financial_advisor")

    # ChromaDB
    chromadb_path: str = os.getenv("CHROMADB_PATH", "./vector_store/chromadb")
    chromadb_host: str = os.getenv("CHROMADB_HOST", "localhost")
    chromadb_port: int = int(os.getenv("CHROMADB_PORT", "8000"))

    # Enhanced LLM Model Settings
    model_name: str = os.getenv("MODEL_NAME", "microsoft/DialoGPT-small")
    model_path: str = os.getenv("MODEL_PATH", "./models/financial_model")
    fine_tune_epochs: int = int(os.getenv("FINE_TUNE_EPOCHS", "3"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "5e-5"))
    max_length: int = int(os.getenv("MAX_LENGTH", "512"))

    # Synthesis Fine-tuning Settings
    synthesis_fine_tuning: bool = os.getenv("SYNTHESIS_FINE_TUNING", "true").lower() == "true"
    synthesis_training_epochs: int = int(os.getenv("SYNTHESIS_TRAINING_EPOCHS", "3"))
    lora_rank: int = int(os.getenv("LORA_RANK", "16"))
    lora_alpha: int = int(os.getenv("LORA_ALPHA", "32"))
    lora_dropout: float = float(os.getenv("LORA_DROPOUT", "0.1"))

    # Enhanced RAG Settings
    query_transformation_enabled: bool = os.getenv("QUERY_TRANSFORMATION", "true").lower() == "true"
    corrective_rag_enabled: bool = os.getenv("CORRECTIVE_RAG", "true").lower() == "true"
    synthesis_quality_threshold: float = float(os.getenv("SYNTHESIS_THRESHOLD", "0.6"))
    max_query_transformations: int = int(os.getenv("MAX_QUERY_TRANSFORMS", "8"))

    # Quality Evaluation Settings
    enable_response_evaluation: bool = os.getenv("ENABLE_EVALUATION", "true").lower() == "true"
    minimum_synthesis_score: float = float(os.getenv("MIN_SYNTHESIS_SCORE", "0.4"))
    quality_score_threshold: float = float(os.getenv("QUALITY_THRESHOLD", "0.7"))

    # Generation Parameters
    generation_temperature: float = float(os.getenv("GENERATION_TEMPERATURE", "0.7"))
    generation_top_p: float = float(os.getenv("GENERATION_TOP_P", "0.9"))
    generation_top_k: int = int(os.getenv("GENERATION_TOP_K", "50"))
    repetition_penalty: float = float(os.getenv("REPETITION_PENALTY", "1.1"))

    # SearXNG
    searxng_url: str = os.getenv("SEARXNG_URL", "http://localhost:8888")

    # Application Settings
    yahoo_finance_enabled: bool = os.getenv("YAHOO_FINANCE_ENABLED", "true").lower() == "true"
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    save_user_data: bool = os.getenv("SAVE_USER_DATA", "true").lower() == "true"
    max_portfolio_size: int = int(os.getenv("MAX_PORTFOLIO_SIZE", "50"))

    # Performance Settings
    enable_gpu_acceleration: bool = os.getenv("ENABLE_GPU", "true").lower() == "true"
    batch_size: int = int(os.getenv("BATCH_SIZE", "2"))
    gradient_accumulation_steps: int = int(os.getenv("GRAD_ACCUMULATION", "4"))

    # Logging and Monitoring
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    enable_performance_monitoring: bool = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
    log_synthesis_quality: bool = os.getenv("LOG_SYNTHESIS_QUALITY", "true").lower() == "true"

# Create global config instance
config = Config()

# Validation functions
def validate_config():
    """Validate configuration settings"""
    issues = []

    if config.synthesis_quality_threshold > 1.0 or config.synthesis_quality_threshold < 0.0:
        issues.append("synthesis_quality_threshold must be between 0.0 and 1.0")

    if config.max_query_transformations > 15:
        issues.append("max_query_transformations should not exceed 15 for performance")

    if config.lora_rank > 64:
        issues.append("lora_rank > 64 may cause memory issues")

    if config.generation_temperature > 2.0 or config.generation_temperature < 0.1:
        issues.append("generation_temperature should be between 0.1 and 2.0")

    if issues:
        print("⚠️ Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("✅ Configuration validation passed")
    return True

def print_config_summary():
    """Print configuration summary"""
    print("🔧 Enhanced Financial Advisor Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Synthesis Fine-tuning: {config.synthesis_fine_tuning}")
    print(f"  Query Transformation: {config.query_transformation_enabled}")
    print(f"  Corrective RAG: {config.corrective_rag_enabled}")
    print(f"  Quality Threshold: {config.synthesis_quality_threshold}")
    print(f"  Max Length: {config.max_length}")
    print(f"  GPU Enabled: {config.enable_gpu_acceleration}")

if __name__ == "__main__":
    validate_config()
    print_config_summary()
