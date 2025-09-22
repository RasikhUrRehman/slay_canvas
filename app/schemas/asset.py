from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel


# Shared properties
class AssetBase(BaseModel):
    type: str                                        # "social", "image", "audio", "document", "text", "wiki", "internet"
    url: Optional[str] = None                        # external links (social, wiki, internet)
    title: Optional[str] = None                      # asset title/name
    content: Optional[str] = None                    # for text assets
    asset_metadata: Optional[Dict[str, Any]] = {}    # flexible extra info
    is_active: Optional[bool] = True


# Input schema for creating an asset
class AssetCreate(AssetBase):
    collection_id: Optional[int] = None              # optional: asset can be in collection


# Input schema for file uploads
class AssetCreateWithFile(BaseModel):
    type: str                                        # "image", "audio", "document"
    title: Optional[str] = None
    asset_metadata: Optional[Dict[str, Any]] = {}
    collection_id: Optional[int] = None
    is_active: Optional[bool] = True


# Input schema for updates
class AssetUpdate(BaseModel):
    type: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    asset_metadata: Optional[Dict[str, Any]] = None
    collection_id: Optional[int] = None
    is_active: Optional[bool] = None


# Response schema
class AssetRead(AssetBase):
    id: int
    file_path: Optional[str] = None                  # MinIO path for uploaded files
    workspace_id: int
    user_id: Optional[int] = None
    collection_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
