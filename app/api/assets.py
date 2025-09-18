from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.asset import AssetCreate, AssetRead
from app.utils.auth import get_current_user_id
from app.services.assets_service import AssetService

router = APIRouter(prefix="/workspaces/{workspace_id}/assets", tags=["assets"])
security = HTTPBearer()
service = AssetService()


# -------------------------
# Social Endpoints
# -------------------------

@router.post("/social/", response_model=AssetRead, status_code=status.HTTP_201_CREATED)
async def create_social_asset(
    workspace_id: int,
    request: AssetCreate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    user_id: int = get_current_user_id(credentials)
    try:
        return await service.create_asset(db, workspace_id, user_id, "social", request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/social/", response_model=List[AssetRead])
async def list_social_assets(
    workspace_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await service.list_assets(db, workspace_id, "social")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/social/{asset_id}", response_model=AssetRead)
async def get_social_asset(
    workspace_id: int,
    asset_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    asset = await service.get_asset(db, workspace_id, asset_id, "social")
    if not asset:
        raise HTTPException(status_code=404, detail="Social asset not found")
    return asset

@router.put("/social/{asset_id}", response_model=AssetRead)
async def update_social_asset(
    workspace_id: int,
    asset_id: int,
    request: AssetCreate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    updated = await service.update_asset(db, workspace_id, asset_id, "social", request)
    if not updated:
        raise HTTPException(status_code=404, detail="Social asset not found")
    return updated


@router.delete("/social/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_social_asset(
    workspace_id: int,
    asset_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    success = await service.delete_asset(db, workspace_id, asset_id, "social")
    if not success:
        raise HTTPException(status_code=404, detail="Social asset not found")
    return None

# -------------------------
# Weblink Endpoints
# -------------------------

@router.post("/weblink/", response_model=AssetRead, status_code=status.HTTP_201_CREATED)
async def create_weblink_asset(
    workspace_id: int,
    request: AssetCreate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    user_id: int = get_current_user_id(credentials)
    try:
        return await service.create_asset(db, workspace_id, user_id, "weblink", request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/weblink/", response_model=List[AssetRead])
async def list_weblink_assets(
    workspace_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await service.list_assets(db, workspace_id, "weblink")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/weblink/{asset_id}", response_model=AssetRead)
async def get_weblink_asset(
    workspace_id: int,
    asset_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    asset = await service.get_asset(db, workspace_id, asset_id, "weblink")
    if not asset:
        raise HTTPException(status_code=404, detail="Weblink asset not found")
    return asset

@router.put("/weblink/{asset_id}", response_model=AssetRead)
async def update_weblink_asset(
    workspace_id: int,
    asset_id: int,
    request: AssetCreate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    updated = await service.update_asset(db, workspace_id, asset_id, "weblink", request)
    if not updated:
        raise HTTPException(status_code=404, detail="Weblink asset not found")
    return updated


@router.delete("/weblink/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_weblink_asset(
    workspace_id: int,
    asset_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    success = await service.delete_asset(db, workspace_id, asset_id, "weblink")
    if not success:
        raise HTTPException(status_code=404, detail="Weblink asset not found")
    return None