"""
Message model for storing chat messages within conversations
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.session import Base
import enum


class MessageRole(enum.Enum):
    USER = "user"
    AGENT = "agent"


class Message(Base):
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True, index=True)
    
    # Message content
    content = Column(Text, nullable=False)
    role = Column(Enum(MessageRole), nullable=False, index=True)
    
    # Conversation relationship
    conversation_id = Column(Integer, ForeignKey('conversations.id'), nullable=False, index=True)
    conversation = relationship("Conversation", back_populates="messages")
    
    # User who sent the message (for user messages)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True, index=True)
    user = relationship("User", backref="messages")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Message(id={self.id}, role='{self.role.value}', conversation_id={self.conversation_id})>"