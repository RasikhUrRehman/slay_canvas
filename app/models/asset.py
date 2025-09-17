from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, ForeignKey, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.session import Base


class Asset(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)

    # Core fields
    type = Column(String, nullable=False, index=True)  # e.g., "social", "image", "document"
    url = Column(Text, nullable=True)                  # store link (social, external, etc.)
    asset_metadata = Column(JSON, default={})                # extra info (title, tags, platform, etc.)
    is_active = Column(Boolean, default=True, nullable=False)

    # Foreign keys (like MongoDB refs)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    # Relationships
    workspace = relationship("Workspace", back_populates="assets")
    user = relationship("User", back_populates="assets")

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Asset(id={self.id}, type='{self.type}', workspace_id={self.workspace_id}, user_id={self.user_id})>"
