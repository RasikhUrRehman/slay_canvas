from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, ForeignKey, Float, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.session import Base


class MediaFile(Base):
    __tablename__ = 'media_files'

    id = Column(Integer, primary_key=True, index=True)
    
    # File information
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String, nullable=False)  # video, audio, image, document
    mime_type = Column(String, nullable=False)
    
    # Processing status
    processing_status = Column(String, default="pending")  # pending, processing, completed, failed
    transcription_status = Column(String, default="pending")
    ai_analysis_status = Column(String, default="pending")
    
    # Media metadata
    duration = Column(Float, nullable=True)  # For video/audio
    dimensions = Column(JSON, nullable=True)  # For images/videos: {"width": 1920, "height": 1080}
    media_metadata = Column(JSON, default={})
    
    # AI-generated content
    transcription_text = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    tags = Column(JSON, default=[])
    ai_insights = Column(JSON, default={})
    
    # Relationships
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    # project_id = Column(Integer, ForeignKey('projects.id'), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="media_files")
    project = relationship("Project", back_populates="media_files")

    def __repr__(self):
        return f"<MediaFile(id={self.id}, filename='{self.filename}')>"


class Project(Base):
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    # Project settings
    settings = Column(JSON, default={})
    canvas_data = Column(JSON, default={})  # Store canvas layout and node positions
    
    # Collaboration
    is_public = Column(Boolean, default=False)
    collaborators = Column(JSON, default=[])  # List of user IDs with permissions
    
    # Relationships
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    workspace_id = Column(Integer, ForeignKey("workspaces.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="projects")
    media_files = relationship("MediaFile", back_populates="project")
    workspace = relationship("Workspace", back_populates="media_files")

    def __repr__(self):
        return f"<Project(id={self.id}, name='{self.name}')>"


# Legacy model for backward compatibility - will be removed
class Media(Base):
    __tablename__ = 'media'

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    path = Column(String, nullable=False)
    media_type = Column(String, nullable=False)  # image|audio|video|document
    uploaded_by = Column(Integer, ForeignKey('users.id'), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
