import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path('.') / '.env')
except Exception:
    pass


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://username:password@localhost:5432/mediaboard_ai"
    POSTGRES_DB: str = "slay_canvas"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    
    # Security
    SECRET_KEY: str = "your-super-secret-key-here-generate-a-random-one"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Google OAuth
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/auth/google/callback"
    
    # Frontend
    FRONTEND_ORIGIN: str = "http://localhost:3000"
    
    # AI Services
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    HF_TOKEN: str = ""
    DEEPGRAM_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = ""
    NLPCLOUD_TOKEN: str = ""
    NLPCLOUD_MODEL: str = ""
    API_NINJAS_KEY: str = ""
    
    # File Storage
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 100000000  # 100MB
    
    # Email Configuration
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # Milvus Configuration
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19531
    MILVUS_COLLECTION: str = "rag_documents"
    
    # Environment
    ENVIRONMENT: str = "development"
    
    class Config:
        env_file = ".env"


settings = Settings()
