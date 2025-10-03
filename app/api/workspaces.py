# # app/api/workspaces.py
# from typing import List, Dict, Any
# from fastapi import APIRouter, Depends, HTTPException, status
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy.future import select

# from sqlalchemy import or_
# from app.db.session import get_db
# from app.services.auth_service import get_current_user
# from app.models.workspace import Workspace as WorkspaceModel
# from app.schemas.workspace import WorkspaceCreate, Workspace, MessageResponse
# from app.models.user import User
# from app.schemas.user import UserPublic
# from app.utils.auth import get_current_user_id
# from app.api.users import get_user_by_id

# router = APIRouter(prefix="/workspaces", tags=["workspaces"])
# security = HTTPBearer()


# @router.post("/", response_model=Workspace, status_code=status.HTTP_201_CREATED)
# async def create_workspace(
#     request: WorkspaceCreate,
#     credentials: HTTPAuthorizationCredentials = Depends(security),
#     db: AsyncSession = Depends(get_db),
# ):
#     """
#     Create a new workspace for the current user.
#     """
#     # Get current authenticated user
#     # print(credentials)
#     user_id: int = get_current_user_id(credentials)
#     # print(user_id)
#     collaborators = []
#     if request.collaborator_ids:
#         collaborators = [
#             await db.get(User, uid) for uid in request.collaborator_ids
#             if await db.get(User, uid)  # skip missing ones
#         ]
#     new_workspace = WorkspaceModel(
#         name=request.name,
#         description=request.description,
#         settings=request.settings or {},
#         is_public=request.is_public,
#         # user_id=user.id,
#         user_id=user_id,
#         users=collaborators,
#     )

#     # Add to DB
#     db.add(new_workspace)
#     await db.flush()  # so we have new_workspace.id available

#     # Attach collaborators (if any)
#     # check = await get_user_by_id(User, 2)
#     # print(check)
#     # if request.collaborator_ids:
#     #     for uid in request.collaborator_ids:
#     #         collaborator = await db.get(User, uid)
#     #         if collaborator:
#     #             new_workspace.users.append(collaborator)

#     await db.commit()
#     await db.refresh(new_workspace)

#     return new_workspace


# @router.get("/", response_model=List[Workspace])
# async def list_workspaces(
#     credentials: HTTPAuthorizationCredentials = Depends(security),
#     db: AsyncSession = Depends(get_db),
# ):
#     """
#     List all workspaces belonging to the current user.
#     """
#     user_id: int = get_current_user_id(credentials)

#     result = await db.execute(
#         # select(WorkspaceModel).where(WorkspaceModel.user_id == user_id)
#         select(WorkspaceModel).where(
#             or_(
#                 WorkspaceModel.user_id == user_id,
#                 WorkspaceModel.users.any(id=user_id)
#             )
#         )
#     )
#     workspaces = result.scalars().all()

#     return workspaces

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.workspace import (
    MessageResponse,
    Workspace,
    WorkspaceCreate,
    WorkspaceUpdate,
)
from app.services.workspace_service import WorkspaceService
from app.utils.auth import get_current_user_id

router = APIRouter(prefix="/workspaces", tags=["workspaces"])
security = HTTPBearer()


@router.post("", response_model=Workspace, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    request: WorkspaceCreate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """Create a new workspace for the current user."""
    user_id: int = get_current_user_id(credentials)
    service = WorkspaceService()
    return await service.create_workspace(db, request, user_id)


@router.get("", response_model=List[Workspace])
async def list_workspaces(
    starred: Optional[bool] = None,
    archived: Optional[bool] = None,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """List workspaces with optional filtering by starred/archived status."""
    user_id: int = get_current_user_id(credentials)
    service = WorkspaceService()
    
    # Handle different filter combinations
    if starred is True and archived is None:
        return await service.list_starred_workspaces(db, user_id)
    elif archived is True and starred is None:
        return await service.list_archived_workspaces(db, user_id)
    elif archived is False and starred is None:
        return await service.list_active_workspaces(db, user_id)
    else:
        # Default: return all workspaces (can be filtered further by combining params)
        workspaces = await service.list_workspaces(db, user_id)
        
        # Apply additional filtering if both params are specified
        if starred is not None:
            workspaces = [w for w in workspaces if w.is_starred == starred]
        if archived is not None:
            workspaces = [w for w in workspaces if w.is_archived == archived]
            
        return workspaces


@router.get("/{workspace_id}", response_model=dict)
async def get_workspace_detailed(
    workspace_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """Get a workspace with all related information (knowledge bases, assets, collections)."""
    user_id: int = get_current_user_id(credentials)
    service = WorkspaceService()
    workspace = await service.get_workspace_detailed(db, workspace_id, user_id)
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Convert to dict to handle the related data properly
    result = {
        "id": workspace.id,
        "name": workspace.name,
        "description": workspace.description,
        "settings": workspace.settings,
        "is_public": workspace.is_public,
        "is_starred": workspace.is_starred,
        "is_archived": workspace.is_archived,
        "user_id": workspace.user_id,
        "created_at": workspace.created_at,
        "updated_at": workspace.updated_at,
        "collaborators": [
            {
                "id": user.id,
                "name": user.name,
                "email": user.email,
            } for user in workspace.users
        ],
        "knowledge_bases": [
            {
                "id": kb.id,
                "name": kb.name,
                "description": kb.description,
                "collection_name": kb.collection_name,
                "is_active": kb.is_active,
                "created_at": kb.created_at,
                "position_x": getattr(kb, 'position_x', 0),
                "position_y": getattr(kb, 'position_y', 0),
                "conversations": [
                    {
                        "id": conv.id,
                        "conversation_name": conv.conversation_name,
                        "user_id": conv.user_id,
                        "created_at": conv.created_at,
                        "updated_at": conv.updated_at,
                    } for conv in kb.conversations
                ],
            } for kb in workspace.knowledge_bases
        ],
        "assets": [
            {
                "id": asset.id,
                "type": asset.type,
                "title": asset.title,
                "url": asset.url,
                "file_path": asset.file_path,
                "content": asset.content,
                "collection_id": asset.collection_id,
                "knowledge_base_id": asset.knowledge_base_id,
                "is_active": asset.is_active,
                "created_at": asset.created_at,
                "position_x": getattr(asset, 'position_x', 0),
                "position_y": getattr(asset, 'position_y', 0),
                "kb_connection_asset_handle": asset.kb_connection_asset_handle,
                "kb_connection_kb_handle": asset.kb_connection_kb_handle,
            } for asset in workspace.assets
        ],
        "collections": [
            {
                "id": collection.id,
                "name": collection.name,
                "description": collection.description,
                "knowledge_base_id": collection.knowledge_base_id,
                "is_active": collection.is_active,
                "created_at": collection.created_at,
                "position_x": getattr(collection, 'position_x', 0),
                "position_y": getattr(collection, 'position_y', 0),
                "asset_count": len([a for a in workspace.assets if a.collection_id == collection.id]),
                "kb_connection_asset_handle": collection.kb_connection_asset_handle,
                "kb_connection_kb_handle": collection.kb_connection_kb_handle,
            } for collection in workspace.collections
        ],
    }
    
    return result


@router.put("/{workspace_id}", response_model=Workspace)
async def update_workspace(
    workspace_id: int,
    request: WorkspaceUpdate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """Update a workspace. Only the owner can update."""
    user_id: int = get_current_user_id(credentials)
    service = WorkspaceService()
    workspace = await service.update_workspace(db, workspace_id, user_id, request)
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found or you don't have permission to update it")
    
    return workspace


@router.delete("/{workspace_id}", response_model=MessageResponse)
async def delete_workspace(
    workspace_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """Delete a workspace. Only the owner can delete."""
    user_id: int = get_current_user_id(credentials)
    service = WorkspaceService()
    success = await service.delete_workspace(db, workspace_id, user_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Workspace not found or you don't have permission to delete it")
    
    return MessageResponse(message=f"Workspace {workspace_id} deleted successfully")


@router.patch("/{workspace_id}/star", response_model=Workspace)
async def star_workspace(
    workspace_id: int,
    is_starred: bool,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """Star or unstar a workspace. Only the owner can star/unstar."""
    user_id: int = get_current_user_id(credentials)
    service = WorkspaceService()
    workspace = await service.star_workspace(db, workspace_id, user_id, is_starred)
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found or you don't have permission to star/unstar it")
    
    return workspace


@router.patch("/{workspace_id}/archive", response_model=Workspace)
async def archive_workspace(
    workspace_id: int,
    is_archived: bool,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """Archive or unarchive a workspace. Only the owner can archive/unarchive."""
    user_id: int = get_current_user_id(credentials)
    service = WorkspaceService()
    workspace = await service.archive_workspace(db, workspace_id, user_id, is_archived)
    
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found or you don't have permission to archive/unarchive it")
    
    return workspace
