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


class Collection(Base):
    __tablename__ = "collections"

    id = Column(Integer, primary_key=True, index=True)

    # Core fields
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    collection_metadata = Column(JSON, default={})  # tags, color, etc.
    is_active = Column(Boolean, default=True, nullable=False)
    
    # React Flow positioning
    position_x = Column(Integer, default=0, nullable=False)  # X coordinate for React Flow
    position_y = Column(Integer, default=0, nullable=False)  # Y coordinate for React Flow

    # Foreign keys
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id", ondelete="SET NULL"), nullable=True, index=True)  # Optional: collection can be linked to KB

    # Relationships
    workspace = relationship("Workspace", back_populates="collections")
    user = relationship("User", back_populates="collections")
    assets = relationship("Asset", back_populates="collection", cascade="all, delete-orphan")
    knowledge_base = relationship("KnowledgeBase", back_populates="collections")

    kb_connection_asset_handle = Column(String, nullable=True)  # Which handle of the asset was used ("left" or "right")
    kb_connection_kb_handle = Column(String, nullable=True)     # Which handle of the KB was used ("left" or "right")

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Collection(id={self.id}, name='{self.name}', workspace_id={self.workspace_id})>"
