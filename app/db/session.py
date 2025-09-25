from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import StaticPool

from app.core.config import settings

# Create async engine
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True if settings.ENVIRONMENT == "development" else False,
    pool_pre_ping=True,
    poolclass=StaticPool if "sqlite" in settings.DATABASE_URL else None,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# Create async session factory
async_session_factory = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Create declarative base
Base = declarative_base()

# Legacy compatibility and exports
engine = async_engine  # For backward compatibility
ENGINE = async_engine
AsyncSessionLocal = async_session_factory
SessionLocal = async_session_factory  # Add this export


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Async session dependency for FastAPI"""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database tables"""
    async with async_engine.begin() as conn:
        # Import all models to ensure they are registered
        from app.models import (
            asset,
            board,
            conversation,
            knowledge_base,
            media,
            otp,
            transcript,
            user,
            workspace,
        )
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connections"""
    await engine.dispose()
