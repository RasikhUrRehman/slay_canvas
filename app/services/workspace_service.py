from typing import List, Optional

from sqlalchemy import and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.models.knowledge_base import KnowledgeBase
from app.models.user import User
from app.models.workspace import Workspace as WorkspaceModel
from app.schemas.workspace import WorkspaceCreate, WorkspaceUpdate


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

    async def get_workspace_by_name(
        self, db: AsyncSession, name: str, user_id: int
    ) -> Optional[WorkspaceModel]:
        """Get a workspace by name for a specific user."""
        result = await db.execute(
            select(WorkspaceModel).where(
                WorkspaceModel.name == name,
                or_(
                    WorkspaceModel.user_id == user_id,
                    WorkspaceModel.users.any(id=user_id),
                )
            )
        )
        return result.scalars().first()

    async def get_workspace_by_id(
        self, db: AsyncSession, workspace_id: int, user_id: int
    ) -> Optional[WorkspaceModel]:
        """Get a workspace by ID for a specific user."""
        result = await db.execute(
            select(WorkspaceModel).where(
                WorkspaceModel.id == workspace_id,
                or_(
                    WorkspaceModel.user_id == user_id,
                    WorkspaceModel.users.any(id=user_id),
                )
            )
        )
        return result.scalars().first()

    async def get_workspace_detailed(
        self, db: AsyncSession, workspace_id: int, user_id: int
    ) -> Optional[WorkspaceModel]:
        """Get a workspace with all related data (knowledge bases, assets, collections)."""
        result = await db.execute(
            select(WorkspaceModel)
            .options(
                selectinload(WorkspaceModel.users),
                selectinload(WorkspaceModel.knowledge_bases).selectinload(KnowledgeBase.conversations),
                selectinload(WorkspaceModel.assets),
                selectinload(WorkspaceModel.collections),
            )
            .where(
                WorkspaceModel.id == workspace_id,
                or_(
                    WorkspaceModel.user_id == user_id,
                    WorkspaceModel.users.any(id=user_id),
                )
            )
        )
        return result.scalars().first()

    async def update_workspace(
        self, db: AsyncSession, workspace_id: int, user_id: int, request: WorkspaceUpdate
    ) -> Optional[WorkspaceModel]:
        """Update a workspace. Only the owner can update."""
        workspace = await db.execute(
            select(WorkspaceModel).where(
                WorkspaceModel.id == workspace_id,
                WorkspaceModel.user_id == user_id  # Only owner can update
            )
        )
        workspace = workspace.scalars().first()
        
        if not workspace:
            return None

        # Update fields if provided
        if request.name is not None:
            workspace.name = request.name
        if request.description is not None:
            workspace.description = request.description
        if request.settings is not None:
            workspace.settings = request.settings
        if request.is_public is not None:
            workspace.is_public = request.is_public
        if request.is_starred is not None:
            workspace.is_starred = request.is_starred
        if request.is_archived is not None:
            workspace.is_archived = request.is_archived
        
        # Update collaborators if provided
        if request.collaborator_ids is not None:
            collaborators = []
            for uid in request.collaborator_ids:
                user = await db.get(User, uid)
                if user:
                    collaborators.append(user)
            workspace.users = collaborators

        await db.commit()
        await db.refresh(workspace)
        return workspace

    async def delete_workspace(
        self, db: AsyncSession, workspace_id: int, user_id: int
    ) -> bool:
        """Delete a workspace. Only the owner can delete."""
        workspace = await db.execute(
            select(WorkspaceModel).where(
                WorkspaceModel.id == workspace_id,
                WorkspaceModel.user_id == user_id  # Only owner can delete
            )
        )
        workspace = workspace.scalars().first()
        
        if not workspace:
            return False

        await db.delete(workspace)
        await db.commit()
        return True

    async def star_workspace(
        self, db: AsyncSession, workspace_id: int, user_id: int, is_starred: bool
    ) -> Optional[WorkspaceModel]:
        """Star or unstar a workspace. Only the owner can star/unstar."""
        workspace = await db.execute(
            select(WorkspaceModel).where(
                WorkspaceModel.id == workspace_id,
                WorkspaceModel.user_id == user_id  # Only owner can star/unstar
            )
        )
        workspace = workspace.scalars().first()
        
        if not workspace:
            return None
        
        workspace.is_starred = is_starred
        await db.commit()
        await db.refresh(workspace)
        return workspace

    async def archive_workspace(
        self, db: AsyncSession, workspace_id: int, user_id: int, is_archived: bool
    ) -> Optional[WorkspaceModel]:
        """Archive or unarchive a workspace. Only the owner can archive/unarchive."""
        workspace = await db.execute(
            select(WorkspaceModel).where(
                WorkspaceModel.id == workspace_id,
                WorkspaceModel.user_id == user_id  # Only owner can archive/unarchive
            )
        )
        workspace = workspace.scalars().first()
        
        if not workspace:
            return None
        
        workspace.is_archived = is_archived
        await db.commit()
        await db.refresh(workspace)
        return workspace

    async def list_starred_workspaces(
        self, db: AsyncSession, user_id: int
    ) -> List[WorkspaceModel]:
        """List all starred workspaces owned by or shared with a user."""
        result = await db.execute(
            select(WorkspaceModel).where(
                and_(
                    WorkspaceModel.is_starred,
                    or_(
                        WorkspaceModel.user_id == user_id,
                        WorkspaceModel.users.any(id=user_id),
                    )
                )
            )
        )
        return result.scalars().all()

    async def list_archived_workspaces(
        self, db: AsyncSession, user_id: int
    ) -> List[WorkspaceModel]:
        """List all archived workspaces owned by or shared with a user."""
        result = await db.execute(
            select(WorkspaceModel).where(
                and_(
                    WorkspaceModel.is_archived,
                    or_(
                        WorkspaceModel.user_id == user_id,
                        WorkspaceModel.users.any(id=user_id),
                    )
                )
            )
        )
        return result.scalars().all()

    async def list_active_workspaces(
        self, db: AsyncSession, user_id: int
    ) -> List[WorkspaceModel]:
        """List all active (non-archived) workspaces owned by or shared with a user."""
        result = await db.execute(
            select(WorkspaceModel).where(
                and_(
                    ~WorkspaceModel.is_archived,
                    or_(
                        WorkspaceModel.user_id == user_id,
                        WorkspaceModel.users.any(id=user_id),
                    )
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
