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


class Asset(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)

    # Core fields
    type = Column(String, nullable=False, index=True)  # "social", "image", "audio", "document", "text", "wiki", "internet", etc.
    url = Column(Text, nullable=True)                  # external links (social, wiki, internet)
    file_path = Column(Text, nullable=True)            # MinIO path for uploaded media files
    title = Column(String, nullable=True)              # asset title/name
    content = Column(Text, nullable=True)              # for text assets
    asset_metadata = Column(JSON, default={})          # extra info (tags, platform, file_size, etc.)
    is_active = Column(Boolean, default=True, nullable=False)

    # Foreign keys
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    collection_id = Column(Integer, ForeignKey("collections.id", ondelete="CASCADE"), nullable=True, index=True)  # Optional: asset can be in collection
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id", ondelete="SET NULL"), nullable=True, index=True)  # Optional: asset can be linked to KB
    
    # React Flow handle tracking for KB connections
    kb_connection_asset_handle = Column(String, nullable=True)  # Which handle of the asset was used ("left" or "right")
    kb_connection_kb_handle = Column(String, nullable=True)     # Which handle of the KB was used ("left" or "right")

    # Relationships
    workspace = relationship("Workspace", back_populates="assets")
    user = relationship("User", back_populates="assets")
    collection = relationship("Collection", back_populates="assets")
    knowledge_base = relationship("KnowledgeBase", back_populates="assets")

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Asset(id={self.id}, type='{self.type}', workspace_id={self.workspace_id}, user_id={self.user_id})>"
