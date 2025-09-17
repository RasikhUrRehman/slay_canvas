# from typing import List, Optional
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy.future import select

# from app.models.asset import Asset as AssetModel
# from app.models.workspace import Workspace as WorkspaceModel
# from app.schemas.asset import AssetCreate


# class AssetService:
#     async def create_asset(
#         self,
#         db: AsyncSession,
#         workspace_id: int,
#         user_id: int,
#         asset_type: str,
#         request: AssetCreate,
#     ) -> AssetModel:
#         """Create an asset of a given type inside a workspace."""
#         workspace = await db.get(WorkspaceModel, workspace_id)
#         if not workspace:
#             raise ValueError("Workspace not found")

#         new_asset = AssetModel(
#             type=asset_type,
#             url=str(request.url) if request.url else None,
#             asset_metadata=request.asset_metadata or {},
#             workspace_id=workspace_id,
#             user_id=user_id,
#         )

#         db.add(new_asset)
#         await db.commit()
#         await db.refresh(new_asset)
#         return new_asset

#     async def list_assets(
#         self, db: AsyncSession, workspace_id: int, asset_type: str
#     ) -> List[AssetModel]:
#         """List all assets of a given type in a workspace."""
#         workspace = await db.get(WorkspaceModel, workspace_id)
#         if not workspace:
#             raise ValueError("Workspace not found")

#         result = await db.execute(
#             select(AssetModel).where(
#                 AssetModel.workspace_id == workspace_id,
#                 AssetModel.type == asset_type,
#             )
#         )
#         return result.scalars().all()

#     async def get_asset(
#         self, db: AsyncSession, workspace_id: int, asset_id: int, asset_type: str
#     ) -> Optional[AssetModel]:
#         """Get a specific asset by ID and type."""
#         asset = await db.get(AssetModel, asset_id)
#         if not asset or asset.workspace_id != workspace_id or asset.type != asset_type:
#             return None
#         return asset

from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.asset import Asset as AssetModel
from app.models.workspace import Workspace as WorkspaceModel
from app.schemas.asset import AssetCreate


class AssetService:
    async def create_asset(
        self,
        db: AsyncSession,
        workspace_id: int,
        user_id: int,
        asset_type: str,
        request: AssetCreate,
    ) -> AssetModel:
        """Create an asset of a given type inside a workspace."""
        workspace = await db.get(WorkspaceModel, workspace_id)
        if not workspace:
            raise ValueError("Workspace not found")

        new_asset = AssetModel(
            type=asset_type,
            url=str(request.url) if request.url else None,
            asset_metadata=request.asset_metadata or {},
            workspace_id=workspace_id,
            user_id=user_id,
        )

        db.add(new_asset)
        await db.commit()
        await db.refresh(new_asset)
        return new_asset

    async def list_assets(
        self, db: AsyncSession, workspace_id: int, asset_type: str
    ) -> List[AssetModel]:
        """List all assets of a given type in a workspace."""
        workspace = await db.get(WorkspaceModel, workspace_id)
        if not workspace:
            raise ValueError("Workspace not found")

        result = await db.execute(
            select(AssetModel).where(
                AssetModel.workspace_id == workspace_id,
                AssetModel.type == asset_type,
            )
        )
        return result.scalars().all()

    async def get_asset(
        self, db: AsyncSession, workspace_id: int, asset_id: int, asset_type: str
    ) -> Optional[AssetModel]:
        """Get a specific asset by ID and type."""
        asset = await db.get(AssetModel, asset_id)
        if not asset or asset.workspace_id != workspace_id or asset.type != asset_type:
            return None
        return asset
