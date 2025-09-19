import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path('.') / '.env')
except Exception:
    pass


class Settings:
    def __init__(self):
        # Database
        self.DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://username:password@localhost:5432/mediaboard_ai")
        self.POSTGRES_DB = os.getenv("POSTGRES_DB", "slay_canvas")
        self.POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
        self.POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
        
        # Security
        self.SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-here-generate-a-random-one")
        self.JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
        self.ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        
        # Google OAuth
        self.GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
        self.GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
        self.GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/api/auth/google/callback")
        
        # Frontend
        self.FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
        
        # AI Services
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
        self.HF_TOKEN = os.getenv("HF_TOKEN", "")
        self.DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
        self.OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "")
        self.NLPCLOUD_TOKEN = os.getenv("NLPCLOUD_TOKEN", "")
        self.NLPCLOUD_MODEL = os.getenv("NLPCLOUD_MODEL", "")
        self.API_NINJAS_KEY = os.getenv("API_NINJAS_KEY", "")
        
        # File Storage
        self.UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
        self.MAX_FILE_SIZE = (os.getenv("MAX_FILE_SIZE", 10000))  # 100MB
        
        # Email Configuration
        self.SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
        self.SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
        self.SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
        
        # Redis
        self.REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        # Vector Database
        self.MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
        self.MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19531"))
        self.MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_documents")
        
        # Environment
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


settings = Settings()
