from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, ForeignKey, Text, Table
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.session import Base

# Association table (no model class needed)
workspace_users = Table(
    "workspace_users",
    Base.metadata,
    Column("workspace_id", Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
)

class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Collaboration and settings
    settings = Column(JSON, default={})
    is_public = Column(Boolean, default=False, nullable=False)

    # Owner relationship
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    user = relationship("User", back_populates="workspaces")

    # Collaborators (many-to-many)
    users = relationship(
        "User",
        secondary=workspace_users,
        back_populates="collaborating_workspaces",
        lazy="selectin"
    )

    # Relationships
    boards = relationship("Board", back_populates="workspace", cascade="all, delete-orphan")
    assets = relationship("Asset", back_populates="workspace", cascade="all, delete-orphan")
    # media_files = relationship("MediaFile", back_populates="workspace", cascade="all, delete-orphan")

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Workspace(id={self.id}, name='{self.name}', user_id={self.user_id})>"
