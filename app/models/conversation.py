"""
Conversation model for managing chat conversations
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.session import Base


class Conversation(Base):
    __tablename__ = 'conversations'

    id = Column(Integer, primary_key=True, index=True)
    conversation_name = Column(String, nullable=False, index=True)
    
    # Project relationship - assuming workspace is the project
    project_id = Column(Integer, ForeignKey('workspaces.id'), nullable=False, index=True)
    project = relationship("Workspace", backref="conversations")
    
    # User who created the conversation
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    user = relationship("User", backref="conversations")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationship to messages
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Conversation(id={self.id}, name='{self.conversation_name}', project_id={self.project_id})>"