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


# Input schema for creating a collection
class CollectionCreate(CollectionBase):
    pass


# Input schema for updates
class CollectionUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    collection_metadata: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


# Response schema
class CollectionRead(CollectionBase):
    id: int
    workspace_id: int
    user_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Collection with assets
class CollectionWithAssets(CollectionRead):
    assets: List[AssetRead] = []
