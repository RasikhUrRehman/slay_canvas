from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.session import Base


class KnowledgeBase(Base):
    """
    Knowledge Base model representing a chatbot with associated Milvus collection
    """
    __tablename__ = 'knowledge_bases'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Milvus collection information
    collection_name = Column(String, unique=True, nullable=False, index=True)
    
    # Configuration
    chunk_size = Column(Integer, default=1000)
    chunk_overlap = Column(Integer, default=200)
    embedding_model = Column(String, default="text-embedding-3-small")
    
    # Metadata and settings
    settings = Column(JSON, default={})
    is_active = Column(Boolean, default=True)
    
    # Owner relationship
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Workspace relationship (optional)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="knowledge_bases")
    workspace = relationship("Workspace", back_populates="knowledge_bases")
    conversations = relationship("Conversation", back_populates="knowledge_base", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<KnowledgeBase(id={self.id}, name='{self.name}', collection='{self.collection_name}')>"

    @property
    def full_collection_name(self):
        """Generate the full Milvus collection name with user prefix"""
        return f"user_{self.user_id}_{self.collection_name}"
