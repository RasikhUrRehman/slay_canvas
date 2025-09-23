"""
Knowledge Base Service for CRUD operations
"""
import logging
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.models.knowledge_base import KnowledgeBase
from app.schemas.knowledge_base import KnowledgeBaseCreate, KnowledgeBaseUpdate
from engine.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    """Service for managing Knowledge Base operations"""
    
    async def create_knowledge_base(
        self, 
        db: AsyncSession, 
        kb_data: KnowledgeBaseCreate, 
        user_id: int
    ) -> KnowledgeBase:
        """Create a new knowledge base and associated Milvus collection"""
        
        # Generate collection name
        collection_name = kb_data.name.lower().replace(' ', '_').replace('-', '_')
        full_collection_name = f"user_{user_id}_{collection_name}"
        
        # Check if collection already exists
        try:
            temp_vector_store = VectorStore(collection_name="temp_check")
            if temp_vector_store.client.has_collection(collection_name=full_collection_name):
                raise ValueError(f"Knowledge base with name '{kb_data.name}' already exists")
        except Exception as e:
            logger.warning(f"Error checking existing collection: {str(e)}")
        
        # Create database record
        kb = KnowledgeBase(
            name=kb_data.name,
            description=kb_data.description,
            collection_name=collection_name,
            chunk_size=kb_data.chunk_size,
            chunk_overlap=kb_data.chunk_overlap,
            embedding_model=kb_data.embedding_model,
            settings=kb_data.settings,
            is_active=kb_data.is_active,
            workspace_id=kb_data.workspace_id,
            user_id=user_id
        )
        
        db.add(kb)
        await db.commit()
        await db.refresh(kb)
        
        # Create Milvus collection
        try:
            vector_store = VectorStore(
                collection_name=full_collection_name,
                dimension=1536  # OpenAI text-embedding-3-small dimension
            )
            logger.info(f"Created Milvus collection: {full_collection_name}")
        except Exception as e:
            # If Milvus creation fails, rollback database
            await db.delete(kb)
            await db.commit()
            raise ValueError(f"Failed to create Milvus collection: {str(e)}")
        
        return kb
    
    async def get_knowledge_base_by_id(
        self, 
        db: AsyncSession, 
        kb_id: int, 
        user_id: int
    ) -> Optional[KnowledgeBase]:
        """Get knowledge base by ID for a specific user"""
        query = select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.user_id == user_id
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_knowledge_base_by_name(
        self, 
        db: AsyncSession, 
        name: str, 
        user_id: int
    ) -> Optional[KnowledgeBase]:
        """Get knowledge base by name for a specific user"""
        query = select(KnowledgeBase).where(
            KnowledgeBase.name == name,
            KnowledgeBase.user_id == user_id
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def list_user_knowledge_bases(
        self, 
        db: AsyncSession, 
        user_id: int,
        workspace_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[KnowledgeBase]:
        """List all knowledge bases for a user"""
        query = select(KnowledgeBase).where(KnowledgeBase.user_id == user_id)
        
        if workspace_id:
            query = query.where(KnowledgeBase.workspace_id == workspace_id)
        
        query = query.offset(skip).limit(limit).order_by(KnowledgeBase.created_at.desc())
        result = await db.execute(query)
        return result.scalars().all()
    
    async def update_knowledge_base(
        self, 
        db: AsyncSession, 
        kb_id: int, 
        kb_update: KnowledgeBaseUpdate, 
        user_id: int
    ) -> Optional[KnowledgeBase]:
        """Update knowledge base"""
        kb = await self.get_knowledge_base_by_id(db, kb_id, user_id)
        if not kb:
            return None
        
        update_data = kb_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(kb, field, value)
        
        await db.commit()
        await db.refresh(kb)
        return kb
    
    async def delete_knowledge_base(
        self, 
        db: AsyncSession, 
        kb_id: int, 
        user_id: int
    ) -> bool:
        """Delete knowledge base and associated Milvus collection"""
        kb = await self.get_knowledge_base_by_id(db, kb_id, user_id)
        if not kb:
            return False
        
        # Delete Milvus collection
        try:
            full_collection_name = kb.full_collection_name
            temp_vector_store = VectorStore(collection_name="temp_check")
            if temp_vector_store.client.has_collection(collection_name=full_collection_name):
                temp_vector_store.client.drop_collection(collection_name=full_collection_name)
                logger.info(f"Deleted Milvus collection: {full_collection_name}")
        except Exception as e:
            logger.error(f"Error deleting Milvus collection: {str(e)}")
            # Continue with database deletion even if Milvus fails
        
        # Delete database record
        await db.delete(kb)
        await db.commit()
        return True
    
    def get_vector_store(self, kb: KnowledgeBase) -> VectorStore:
        """Get VectorStore instance for a knowledge base"""
        return VectorStore(collection_name=kb.full_collection_name)
    
    async def get_knowledge_base_stats(
        self, 
        kb: KnowledgeBase
    ) -> dict:
        """Get statistics for a knowledge base from Milvus"""
        try:
            vector_store = self.get_vector_store(kb)
            stats = vector_store.get_collection_stats()
            documents = vector_store.list_all_documents()
            
            # Count unique documents by source_url
            unique_sources = set()
            for _, metadata in documents:
                source_url = metadata.get('source_url', '')
                if source_url:
                    unique_sources.add(source_url)
            
            return {
                "document_count": len(unique_sources),
                "chunk_count": stats.get("total_entities", 0),
                "milvus_stats": stats
            }
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {str(e)}")
            return {
                "document_count": 0,
                "chunk_count": 0,
                "milvus_stats": {"error": str(e)}
            }


# Global service instance
knowledge_base_service = KnowledgeBaseService()
