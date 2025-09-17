from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import or_

from app.models.workspace import Workspace as WorkspaceModel
from app.models.user import User
from app.schemas.workspace import WorkspaceCreate


class WorkspaceService:
    async def create_workspace(
        self, db: AsyncSession, request: WorkspaceCreate, user_id: int
    ) -> WorkspaceModel:
        """Create a new workspace for a user with optional collaborators."""
        collaborators: List[User] = []
        if request.collaborator_ids:
            collaborators = [
                await db.get(User, uid) for uid in request.collaborator_ids
                if await db.get(User, uid)
            ]

        new_workspace = WorkspaceModel(
            name=request.name,
            description=request.description,
            settings=request.settings or {},
            is_public=request.is_public,
            user_id=user_id,
            users=collaborators,
        )

        db.add(new_workspace)
        await db.flush()   # assign ID
        await db.commit()
        await db.refresh(new_workspace)

        return new_workspace

    async def list_workspaces(
        self, db: AsyncSession, user_id: int
    ) -> List[WorkspaceModel]:
        """List all workspaces owned by or shared with a user."""
        result = await db.execute(
            select(WorkspaceModel).where(
                or_(
                    WorkspaceModel.user_id == user_id,
                    WorkspaceModel.users.any(id=user_id),
                )
            )
        )
        return result.scalars().all()


# Legacy function for backward compatibility
async def create_workspace_service(
    request: WorkspaceCreate, user_id: int, db: AsyncSession
) -> WorkspaceModel:
    service = WorkspaceService()
    return await service.create_workspace(db, request, user_id)


async def list_workspaces_service(
    user_id: int, db: AsyncSession
) -> List[WorkspaceModel]:
    service = WorkspaceService()
    return await service.list_workspaces(db, user_id)


# from typing import List, Optional
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy.future import select
# from sqlalchemy import or_

# from app.models.workspace import Workspace as WorkspaceModel
# from app.models.user import User
# from app.schemas.workspace import WorkspaceCreate


# async def create_workspace_service(
#     request: WorkspaceCreate, user_id: int, db: AsyncSession
# ) -> WorkspaceModel:
#     """
#     Create a new workspace for the given user.
#     """
#     # Collect collaborators
#     collaborators = []
#     if request.collaborator_ids:
#         collaborators = [
#             await db.get(User, uid) for uid in request.collaborator_ids
#             if await db.get(User, uid)
#         ]

#     new_workspace = WorkspaceModel(
#         name=request.name,
#         description=request.description,
#         settings=request.settings or {},
#         is_public=request.is_public,
#         user_id=user_id,
#         users=collaborators,
#     )

#     db.add(new_workspace)
#     await db.flush()   # Get id
#     await db.commit()
#     await db.refresh(new_workspace)

#     return new_workspace


# async def list_workspaces_service(user_id: int, db: AsyncSession) -> List[WorkspaceModel]:
#     """
#     Get all workspaces for a user (owned or as collaborator).
#     """
#     result = await db.execute(
#         select(WorkspaceModel).where(
#             or_(
#                 WorkspaceModel.user_id == user_id,
#                 WorkspaceModel.users.any(id=user_id)
#             )
#         )
#     )
#     return result.scalars().all()
