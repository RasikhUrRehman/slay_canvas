from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.session import Base
from app.models.workspace import workspace_users

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)  # Changed from full_name to name
    hashed_password = Column(String, nullable=True)  # Nullable for OAuth users
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_superuser = Column(Boolean, default=False)
    
    # OAuth fields
    google_id = Column(String, unique=True, nullable=True)
    avatar_url = Column(String, nullable=True)
    provider = Column(String, nullable=True)  # e.g., 'google', 'email'
    
    # Profile data
    profile_data = Column(JSON, default={})
    
    # Subscription info
    subscription_plan = Column(String, default="free")
    subscription_status = Column(String, default="active")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    boards = relationship("Board", back_populates="user", cascade="all, delete-orphan")
    workspaces = relationship("Workspace", back_populates="user", cascade="all, delete-orphan")
    collaborating_workspaces = relationship(
        "Workspace",
        secondary=workspace_users,
        back_populates="users"
    )
    assets = relationship("Asset", back_populates="user", cascade="all, delete-orphan")
    # projects = relationship("Project", back_populates="user")
    # media_files = relationship("MediaFile", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}')>"
