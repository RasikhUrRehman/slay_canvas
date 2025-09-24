from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

# Import the actual schemas to avoid forward reference issues
from app.schemas.asset import AssetRead
from app.schemas.collection import CollectionRead
from app.schemas.knowledge_base import KnowledgeBasePublic
from app.schemas.user import UserPublic


# ✅ Shared properties
class WorkspaceBase(BaseModel):
    name: str
    description: Optional[str] = None
    settings: Dict[str, Any] = {}
    is_public: bool = False


# ✅ Schema for creating a workspace
class WorkspaceCreate(WorkspaceBase):
    collaborator_ids: Optional[List[int]] = []  # just IDs of users to add as collaborators


# ✅ Schema for updating a workspace (all optional)
class WorkspaceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    is_public: Optional[bool] = None
    collaborator_ids: Optional[List[int]] = None


# ✅ Base schema with DB fields
class WorkspaceInDBBase(WorkspaceBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ✅ Schema for internal DB usage
class WorkspaceInDB(WorkspaceInDBBase):
    pass


# ✅ Public workspace schema (API responses)
class Workspace(WorkspaceInDBBase):
    collaborators: List[UserPublic] = []  # return list of user objects


# ✅ A lighter public version (if you don't want to expose collaborators/settings)
class WorkspacePublic(BaseModel):
    id: int
    name: str
    description: Optional[str]
    is_public: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ✅ Comprehensive workspace with all related data
class WorkspaceDetailed(WorkspaceInDBBase):
    collaborators: List[UserPublic] = []
    knowledge_bases: List[KnowledgeBasePublic] = []
    assets: List[AssetRead] = []
    collections: List[CollectionRead] = []


# ✅ Response message (useful for success/failure messages)
class MessageResponse(BaseModel):
    message: str
