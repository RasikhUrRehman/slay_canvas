"""
Board model for organizing research and creative resources
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.session import Base


class Board(Base):
    __tablename__ = 'boards'

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Privacy and access control
    is_private = Column(Boolean, default=True, nullable=False)
    
    # Owner relationship
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    user = relationship("User", back_populates="boards")
    
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=False)
    workspace = relationship("Workspace", back_populates="boards")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Future relationships for board items/resources
    # board_items = relationship("BoardItem", back_populates="board", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Board(id={self.id}, title='{self.title}', user_id={self.user_id})>"
