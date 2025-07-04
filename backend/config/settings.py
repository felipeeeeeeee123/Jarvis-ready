"""Centralized configuration settings for JARVIS v3.0."""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
BLUEPRINTS_DIR = BASE_DIR / "blueprints"
SIMULATIONS_DIR = BASE_DIR / "simulations"
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"

# Create directories
for dir_path in [DATA_DIR, LOGS_DIR, BLUEPRINTS_DIR, SIMULATIONS_DIR, MODELS_DIR, UPLOADS_DIR]:
    dir_path.mkdir(exist_ok=True)


class Settings:
    """Application settings."""
    
    # === Application ===
    APP_NAME: str = "JARVIS v3.0"
    APP_VERSION: str = "3.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # === API Authentication ===
    API_KEY: Optional[str] = os.getenv("JARVIS_API_KEY")
    
    # === Database ===
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/jarvis.db")
    
    # === Alpaca Trading ===
    ALPACA_API_KEY: Optional[str] = os.getenv("APCA_API_KEY_ID")
    ALPACA_SECRET_KEY: Optional[str] = os.getenv("APCA_API_SECRET_KEY")
    ALPACA_BASE_URL: str = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    
    # === Trading Configuration ===
    TRADE_PERCENT: float = float(os.getenv("TRADE_PERCENT", "0.05"))
    TRADE_CAP: float = float(os.getenv("TRADE_CAP", "40"))
    STRATEGY: str = os.getenv("STRATEGY", "RSI").upper()
    TRADE_COOLDOWN: int = int(os.getenv("TRADE_COOLDOWN", "3600"))
    
    # === Telegram ===
    TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    
    # === OpenAI ===
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # === Ollama ===
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "localhost")
    OLLAMA_PORT: int = int(os.getenv("OLLAMA_PORT", "11434"))
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
    
    # === Logging ===
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_TO_FILE: bool = os.getenv("LOG_TO_FILE", "true").lower() == "true"
    LOG_TO_CONSOLE: bool = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
    
    # === Web Search ===
    WEB_SEARCH_ENABLED: bool = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"
    WEB_SEARCH_RESULTS_LIMIT: int = int(os.getenv("WEB_SEARCH_RESULTS_LIMIT", "3"))
    
    # === Flask Server ===
    FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "8000"))
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    
    # === Performance ===
    MAX_MEMORY_ENTRIES: int = int(os.getenv("MAX_MEMORY_ENTRIES", "10000"))
    MAX_QA_ENTRIES: int = int(os.getenv("MAX_QA_ENTRIES", "1000"))
    AUTOTRAIN_ENABLED: bool = os.getenv("AUTOTRAIN_ENABLED", "true").lower() == "true"
    
    # === Security ===
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-key-change-in-production")
    ALLOWED_HOSTS: List[str] = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    
    # === Backup ===
    BACKUP_ENABLED: bool = os.getenv("BACKUP_ENABLED", "true").lower() == "true"
    BACKUP_INTERVAL_HOURS: int = int(os.getenv("BACKUP_INTERVAL_HOURS", "12"))
    BACKUP_RETENTION_DAYS: int = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
    
    @classmethod
    def validate(cls) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Required for trading
        if cls.ALPACA_API_KEY and not cls.ALPACA_SECRET_KEY:
            errors.append("ALPACA_SECRET_KEY is required when ALPACA_API_KEY is set")
        
        if cls.ALPACA_SECRET_KEY and not cls.ALPACA_API_KEY:
            errors.append("ALPACA_API_KEY is required when ALPACA_SECRET_KEY is set")
        
        # Trading limits
        if cls.TRADE_PERCENT <= 0 or cls.TRADE_PERCENT > 1:
            errors.append("TRADE_PERCENT must be between 0 and 1")
        
        if cls.TRADE_CAP <= 0:
            errors.append("TRADE_CAP must be positive")
        
        if cls.STRATEGY not in ["RSI", "EMA", "MACD"]:
            errors.append("STRATEGY must be one of: RSI, EMA, MACD")
        
        # Telegram validation
        if cls.TELEGRAM_BOT_TOKEN and not cls.TELEGRAM_CHAT_ID:
            errors.append("TELEGRAM_CHAT_ID is required when TELEGRAM_BOT_TOKEN is set")
        
        return errors
    
    @classmethod
    def to_dict(cls) -> dict:
        """Convert settings to dictionary (excluding sensitive data)."""
        sensitive_keys = {"API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", 
                         "TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY", "SECRET_KEY"}
        
        return {
            key: getattr(cls, key) 
            for key in dir(cls) 
            if not key.startswith("_") and not callable(getattr(cls, key))
            and key not in sensitive_keys
        }


# Global settings instance
settings = Settings()

# Validate configuration on import
config_errors = settings.validate()
if config_errors:
    print("Configuration errors found:")
    for error in config_errors:
        print(f"  - {error}")
    print("Please check your environment variables and .env file")