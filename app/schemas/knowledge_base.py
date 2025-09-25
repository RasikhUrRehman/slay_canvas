"""
Pydantic schemas for Knowledge Base operations
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class KnowledgeBaseBase(BaseModel):
    """Base schema for Knowledge Base"""
    name: str
    description: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "text-embedding-3-small"
    settings: Dict[str, Any] = {}
    is_active: bool = True
    workspace_id: Optional[int] = None


class KnowledgeBaseCreate(KnowledgeBaseBase):
    """Schema for creating a Knowledge Base"""
    pass


class KnowledgeBaseUpdate(BaseModel):
    """Schema for updating a Knowledge Base"""
    name: Optional[str] = None
    description: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    embedding_model: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    workspace_id: Optional[int] = None


class KnowledgeBasePublic(KnowledgeBaseBase):
    """Public schema for Knowledge Base responses"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    collection_name: str
    user_id: int
    created_at: datetime
    updated_at: datetime
    
    # Computed fields
    full_collection_name: str
    
    # Optional relationship data
    document_count: Optional[int] = None
    chunk_count: Optional[int] = None


class KnowledgeBaseWithStats(KnowledgeBasePublic):
    """Knowledge Base with additional statistics"""
    document_count: int
    chunk_count: int
    last_updated: Optional[datetime] = None
    milvus_stats: Dict[str, Any] = {}


class KnowledgeBaseListResponse(BaseModel):
    """Response for listing knowledge bases"""
    knowledge_bases: List[KnowledgeBaseWithStats]
    total_count: int
    page: int = 1
    per_page: int = 10


# Legacy compatibility schemas for the agent API
class CreateKnowledgeBaseRequest(BaseModel):
    """Legacy schema for backward compatibility"""
    name: str
    project_name: str
    description: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200


class KnowledgeBaseInfo(BaseModel):
    """Legacy schema for backward compatibility"""
    name: str
    description: Optional[str]
    document_count: int
    chunk_count: int
    created_at: str
    stats: Dict[str, Any]
