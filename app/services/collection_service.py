from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.models.collection import Collection
from app.models.workspace import Workspace
from app.schemas.collection import CollectionCreate, CollectionUpdate


class CollectionService:
    async def create_collection(
        self, 
        db: AsyncSession, 
        workspace_id: int, 
        user_id: int, 
        collection_data: CollectionCreate
    ) -> Collection:
        """Create a new collection in a workspace."""
        # Verify workspace exists and user has access
        workspace = await db.get(Workspace, workspace_id)
        if not workspace:
            raise ValueError("Workspace not found")
        
        # Create collection
        collection = Collection(
            name=collection_data.name,
            description=collection_data.description,
            collection_metadata=collection_data.collection_metadata or {},
            is_active=collection_data.is_active,
            workspace_id=workspace_id,
            user_id=user_id
        )
        
        db.add(collection)
        await db.commit()
        await db.refresh(collection)
        return collection

    async def get_collection(
        self, 
        db: AsyncSession, 
        workspace_id: int, 
        collection_id: int,
        include_assets: bool = False
    ) -> Optional[Collection]:
        """Get a collection by ID within a workspace."""
        query = select(Collection).where(
            Collection.id == collection_id,
            Collection.workspace_id == workspace_id,
            Collection.is_active
        )
        
        if include_assets:
            query = query.options(selectinload(Collection.assets))
        
        result = await db.execute(query)
        return result.scalars().first()

    async def list_collections(
        self, 
        db: AsyncSession, 
        workspace_id: int,
        include_assets: bool = False
    ) -> List[Collection]:
        """List all collections in a workspace."""
        query = select(Collection).where(
            Collection.workspace_id == workspace_id,
            Collection.is_active
        ).order_by(Collection.created_at.desc())
        
        if include_assets:
            query = query.options(selectinload(Collection.assets))
        
        result = await db.execute(query)
        return result.scalars().all()

    async def update_collection(
        self, 
        db: AsyncSession, 
        workspace_id: int, 
        collection_id: int, 
        collection_data: CollectionUpdate
    ) -> Optional[Collection]:
        """Update a collection."""
        collection = await self.get_collection(db, workspace_id, collection_id)
        if not collection:
            return None
        
        # Update fields
        if collection_data.name is not None:
            collection.name = collection_data.name
        if collection_data.description is not None:
            collection.description = collection_data.description
        if collection_data.collection_metadata is not None:
            collection.collection_metadata = collection_data.collection_metadata
        if collection_data.is_active is not None:
            collection.is_active = collection_data.is_active
        
        await db.commit()
        await db.refresh(collection)
        return collection

    async def delete_collection(
        self, 
        db: AsyncSession, 
        workspace_id: int, 
        collection_id: int
    ) -> bool:
        """Delete a collection (soft delete)."""
        collection = await self.get_collection(db, workspace_id, collection_id)
        if not collection:
            return False
        
        collection.is_active = False
        await db.commit()
        return True

    async def hard_delete_collection(
        self, 
        db: AsyncSession, 
        workspace_id: int, 
        collection_id: int
    ) -> bool:
        """Permanently delete a collection and all its assets."""
        collection = await self.get_collection(db, workspace_id, collection_id, include_assets=True)
        if not collection:
            return False
        
        await db.delete(collection)
        await db.commit()
        return True
