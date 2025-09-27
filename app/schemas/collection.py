from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from app.schemas.asset import AssetRead


# Shared properties
class CollectionBase(BaseModel):
    name: str
    description: Optional[str] = None
    collection_metadata: Optional[Dict[str, Any]] = {}
    is_active: Optional[bool] = True
    position_x: Optional[int] = 0                    # React Flow X position
    position_y: Optional[int] = 0                    # React Flow Y position


# Input schema for creating a collection
class CollectionCreate(CollectionBase):
    pass


# Input schema for updates
class CollectionUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    collection_metadata: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    position_x: Optional[int] = None                 # React Flow X position
    position_y: Optional[int] = None                 # React Flow Y position


# Response schema
class CollectionRead(CollectionBase):
    id: int
    workspace_id: int
    user_id: Optional[int] = None
    knowledge_base_id: Optional[int] = None          # linked knowledge base
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Collection with assets
class CollectionWithAssets(CollectionRead):
    assets: List[AssetRead] = []
