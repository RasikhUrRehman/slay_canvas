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

from typing import List
from fastapi import APIRouter, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.workspace import WorkspaceCreate, Workspace
from app.utils.auth import get_current_user_id
from app.services.workspace_service import WorkspaceService

router = APIRouter(prefix="/workspaces", tags=["workspaces"])
security = HTTPBearer()


@router.post("/", response_model=Workspace, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    request: WorkspaceCreate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    user_id: int = get_current_user_id(credentials)
    service = WorkspaceService()
    return await service.create_workspace(db, request, user_id)


@router.get("/", response_model=List[Workspace])
async def list_workspaces(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    user_id: int = get_current_user_id(credentials)
    service = WorkspaceService()
    return await service.list_workspaces(db, user_id)
