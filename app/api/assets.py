# from typing import List
# from fastapi import APIRouter, Depends, HTTPException, status
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy.future import select

# from app.db.session import get_db
# from app.models.asset import Asset as AssetModel
# from app.models.workspace import Workspace as WorkspaceModel
# from app.schemas.asset import AssetCreate, AssetRead
# from app.utils.auth import get_current_user_id

# router = APIRouter(prefix="/workspaces/{workspace_id}/assets", tags=["assets"])
# security = HTTPBearer()

# # -------------------------
# # Social Media Endpoints
# # -------------------------

# @router.post("/social/", response_model=AssetRead, status_code=status.HTTP_201_CREATED)
# async def create_social_asset(
#     workspace_id: int,
#     request: AssetCreate,
#     credentials: HTTPAuthorizationCredentials = Depends(security),
#     db: AsyncSession = Depends(get_db),
# ):
#     """
#     Upload a new social media link to a workspace.
#     """
#     user_id: int = get_current_user_id(credentials)

#     workspace = await db.get(WorkspaceModel, workspace_id)
#     if not workspace:
#         raise HTTPException(status_code=404, detail="Workspace not found")

#     new_asset = AssetModel(
#         type="social",
#         url=str(request.url) if request.url else None,
#         asset_metadata=request.asset_metadata or {},
#         workspace_id=workspace_id,
#         user_id=user_id,
#     )

#     db.add(new_asset)
#     await db.commit()
#     await db.refresh(new_asset)

#     return new_asset


# @router.get("/social/", response_model=List[AssetRead])
# async def list_social_assets(
#     workspace_id: int,
#     credentials: HTTPAuthorizationCredentials = Depends(security),
#     db: AsyncSession = Depends(get_db),
# ):
#     """
#     Get all social media links for a workspace.
#     """
#     user_id: int = get_current_user_id(credentials)

#     workspace = await db.get(WorkspaceModel, workspace_id)
#     if not workspace:
#         raise HTTPException(status_code=404, detail="Workspace not found")

#     result = await db.execute(
#         select(AssetModel).where(
#             AssetModel.workspace_id == workspace_id,
#             AssetModel.type == "social"
#         )
#     )
#     return result.scalars().all()


# @router.get("/social/{asset_id}", response_model=AssetRead)
# async def get_social_asset(
#     workspace_id: int,
#     asset_id: int,
#     credentials: HTTPAuthorizationCredentials = Depends(security),
#     db: AsyncSession = Depends(get_db),
# ):
#     """
#     Get a specific social media link by ID.
#     """
#     user_id: int = get_current_user_id(credentials)

#     asset = await db.get(AssetModel, asset_id)
#     if not asset or asset.workspace_id != workspace_id or asset.type != "social":
#         raise HTTPException(status_code=404, detail="Social asset not found")

#     return asset


# # -------------------------
# # Weblink Endpoints
# # -------------------------

# @router.post("/weblink/", response_model=AssetRead, status_code=status.HTTP_201_CREATED)
# async def create_weblink_asset(
#     workspace_id: int,
#     request: AssetCreate,
#     credentials: HTTPAuthorizationCredentials = Depends(security),
#     db: AsyncSession = Depends(get_db),
# ):
#     """
#     Upload a new weblink to a workspace.
#     """
#     user_id: int = get_current_user_id(credentials)

#     workspace = await db.get(WorkspaceModel, workspace_id)
#     if not workspace:
#         raise HTTPException(status_code=404, detail="Workspace not found")

#     new_asset = AssetModel(
#         type="weblink",
#         url=str(request.url) if request.url else None,
#         asset_metadata=request.asset_metadata or {},
#         workspace_id=workspace_id,
#         user_id=user_id,
#     )

#     db.add(new_asset)
#     await db.commit()
#     await db.refresh(new_asset)

#     return new_asset


# @router.get("/weblink/", response_model=List[AssetRead])
# async def list_weblink_assets(
#     workspace_id: int,
#     credentials: HTTPAuthorizationCredentials = Depends(security),
#     db: AsyncSession = Depends(get_db),
# ):
#     """
#     Get all weblinks for a workspace.
#     """
#     user_id: int = get_current_user_id(credentials)

#     workspace = await db.get(WorkspaceModel, workspace_id)
#     if not workspace:
#         raise HTTPException(status_code=404, detail="Workspace not found")

#     result = await db.execute(
#         select(AssetModel).where(
#             AssetModel.workspace_id == workspace_id,
#             AssetModel.type == "weblink"
#         )
#     )
#     return result.scalars().all()


# @router.get("/weblink/{asset_id}", response_model=AssetRead)
# async def get_weblink_asset(
#     workspace_id: int,
#     asset_id: int,
#     credentials: HTTPAuthorizationCredentials = Depends(security),
#     db: AsyncSession = Depends(get_db),
# ):
#     """
#     Get a specific weblink by ID.
#     """
#     user_id: int = get_current_user_id(credentials)

#     asset = await db.get(AssetModel, asset_id)
#     if not asset or asset.workspace_id != workspace_id or asset.type != "weblink":
#         raise HTTPException(status_code=404, detail="Weblink asset not found")

#     return asset


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
