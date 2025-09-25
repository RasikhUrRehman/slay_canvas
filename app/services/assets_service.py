from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.asset import Asset as AssetModel
from app.models.collection import Collection as CollectionModel
from app.models.workspace import Workspace as WorkspaceModel
from app.schemas.asset import AssetCreate, AssetCreateWithFile


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

        # Validate collection if provided
        collection_id = None
        if request.collection_id and request.collection_id != 0:
            collection = await db.get(CollectionModel, request.collection_id)
            if not collection or collection.workspace_id != workspace_id:
                raise ValueError("Collection not found or doesn't belong to this workspace")
            collection_id = request.collection_id

        new_asset = AssetModel(
            type=asset_type,
            url=request.url,
            title=request.title,
            content=request.content,
            asset_metadata=request.asset_metadata or {},
            collection_id=collection_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )

        db.add(new_asset)
        await db.commit()
        await db.refresh(new_asset)
        return new_asset

    async def create_file_asset(
        self,
        db: AsyncSession,
        workspace_id: int,
        user_id: int,
        request: AssetCreateWithFile,
        file_path: str,
    ) -> AssetModel:
        """Create a file-based asset with MinIO path."""
        workspace = await db.get(WorkspaceModel, workspace_id)
        if not workspace:
            raise ValueError("Workspace not found")

        # Validate collection if provided
        collection_id = None
        if request.collection_id and request.collection_id != 0:
            collection = await db.get(CollectionModel, request.collection_id)
            if not collection or collection.workspace_id != workspace_id:
                raise ValueError("Collection not found or doesn't belong to this workspace")
            collection_id = request.collection_id

        new_asset = AssetModel(
            type=request.type,
            title=request.title,
            file_path=file_path,
            asset_metadata=request.asset_metadata or {},
            collection_id=collection_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )

        db.add(new_asset)
        await db.commit()
        await db.refresh(new_asset)
        return new_asset

    async def list_assets(
        self, 
        db: AsyncSession, 
        workspace_id: int, 
        asset_type: Optional[str] = None,
        collection_id: Optional[int] = None
    ) -> List[AssetModel]:
        """List assets in a workspace, optionally filtered by type or collection."""
        workspace = await db.get(WorkspaceModel, workspace_id)
        if not workspace:
            raise ValueError("Workspace not found")

        query = select(AssetModel).where(AssetModel.workspace_id == workspace_id)
        
        if asset_type:
            query = query.where(AssetModel.type == asset_type)
        
        if collection_id:
            query = query.where(AssetModel.collection_id == collection_id)
        
        result = await db.execute(query.order_by(AssetModel.created_at.desc()))
        return result.scalars().all()

    async def get_asset(
        self, db: AsyncSession, workspace_id: int, asset_id: int
    ) -> Optional[AssetModel]:
        """Get a specific asset by ID."""
        asset = await db.get(AssetModel, asset_id)
        if not asset or asset.workspace_id != workspace_id:
            return None
        return asset

    async def update_asset(
        self,
        db: AsyncSession,
        workspace_id: int,
        asset_id: int,
        request: AssetCreate,
    ) -> Optional[AssetModel]:
        """Update an asset."""
        asset = await self.get_asset(db, workspace_id, asset_id)
        if not asset:
            return None

        # Validate collection if provided
        collection_id = asset.collection_id  # Keep existing value by default
        if request.collection_id is not None:
            if request.collection_id == 0:
                collection_id = None  # Explicitly set to None if 0 is passed
            else:
                collection = await db.get(CollectionModel, request.collection_id)
                if not collection or collection.workspace_id != workspace_id:
                    raise ValueError("Collection not found or doesn't belong to this workspace")
                collection_id = request.collection_id

        # Update fields
        if request.url is not None:
            asset.url = request.url
        if request.title is not None:
            asset.title = request.title
        if request.content is not None:
            asset.content = request.content
        if request.asset_metadata is not None:
            asset.asset_metadata = request.asset_metadata
        
        # Apply the validated collection_id
        asset.collection_id = collection_id

        await db.commit()
        await db.refresh(asset)
        return asset

    async def delete_asset(
        self, db: AsyncSession, workspace_id: int, asset_id: int
    ) -> bool:
        """Delete an asset."""
        asset = await self.get_asset(db, workspace_id, asset_id)
        if not asset:
            return False

        await db.delete(asset)
        await db.commit()
        return True

    async def link_asset_to_collection(
        self,
        db: AsyncSession,
        workspace_id: int,
        asset_id: int,
        collection_id: int,
        user_id: int,
    ) -> dict:
        """Link an existing asset to a collection."""
        # Get the asset
        asset = await self.get_asset(db, workspace_id, asset_id)
        if not asset:
            raise ValueError("Asset not found")
        
        # Verify asset ownership
        if asset.user_id != user_id:
            raise ValueError("Access denied: You don't own this asset")
        
        # Get the collection
        collection = await db.get(CollectionModel, collection_id)
        if not collection or collection.workspace_id != workspace_id:
            raise ValueError("Collection not found or doesn't belong to this workspace")
        
        # Verify collection ownership
        if collection.user_id != user_id:
            raise ValueError("Access denied: You don't own this collection")
        
        # Check if asset is already linked to this collection
        if asset.collection_id == collection_id:
            raise ValueError("Asset is already linked to this collection")
        
        # Link the asset to the collection
        old_collection_id = asset.collection_id
        asset.collection_id = collection_id
        
        await db.commit()
        await db.refresh(asset)
        
        return {
            "message": f"Asset '{asset.title or asset.id}' linked to collection '{collection.name}'",
            "asset_id": asset.id,
            "collection_id": collection_id,
            "previous_collection_id": old_collection_id
        }

    async def unlink_asset_from_collection(
        self,
        db: AsyncSession,
        workspace_id: int,
        asset_id: int,
        user_id: int,
    ) -> dict:
        """Unlink an asset from its collection."""
        # Get the asset
        asset = await self.get_asset(db, workspace_id, asset_id)
        if not asset:
            raise ValueError("Asset not found")
        
        # Verify asset ownership
        if asset.user_id != user_id:
            raise ValueError("Access denied: You don't own this asset")
        
        # Check if asset is linked to a collection
        if not asset.collection_id:
            raise ValueError("Asset is not linked to any collection")
        
        # Get collection info for response message
        collection = await db.get(CollectionModel, asset.collection_id)
        collection_name = collection.name if collection else "Unknown Collection"
        old_collection_id = asset.collection_id
        
        # Unlink from collection
        asset.collection_id = None
        
        await db.commit()
        await db.refresh(asset)
        
        return {
            "message": f"Asset '{asset.title or asset.id}' unlinked from collection '{collection_name}'",
            "asset_id": asset.id,
            "previous_collection_id": old_collection_id
        }