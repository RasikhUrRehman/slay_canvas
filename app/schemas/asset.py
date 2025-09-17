from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
from datetime import datetime


# Shared properties
class AssetBase(BaseModel):
    url: Optional[HttpUrl] = None                # link (social, external, etc.)
    asset_metadata: Optional[Dict[str, Any]] = {}  # flexible extra info
    is_active: Optional[bool] = True


# Input schema for creating a social asset
class AssetCreate(AssetBase):
    pass   # workspace_id comes from path, type is forced = "social"


# Input schema for updates
class AssetUpdate(BaseModel):
    url: Optional[HttpUrl] = None
    asset_metadata: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


# Response schema
class AssetRead(AssetBase):
    id: int
    type: str
    workspace_id: int
    user_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
