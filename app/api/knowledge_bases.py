"""
Knowledge Bases API endpoints
"""
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.knowledge_base import (
    KnowledgeBaseCreate,
    KnowledgeBasePublic,
    KnowledgeBaseUpdate,
)
from app.services.knowledge_base_service import knowledge_base_service
from app.utils.auth import get_current_user_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])
security = HTTPBearer()


@router.post("/", response_model=KnowledgeBasePublic, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    request: KnowledgeBaseCreate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """Create a new knowledge base."""
    user_id: int = get_current_user_id(credentials)
    return await knowledge_base_service.create_knowledge_base(db, request, user_id)


@router.get("/", response_model=List[KnowledgeBasePublic])
async def list_knowledge_bases(
    workspace_id: int = None,
    skip: int = 0,
    limit: int = 100,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """List knowledge bases for the current user."""
    user_id: int = get_current_user_id(credentials)
    knowledge_bases = await knowledge_base_service.list_user_knowledge_bases(
        db, user_id, workspace_id, skip, limit
    )
    
    # Enrich each knowledge base with stats
    enriched_kbs = []
    for kb in knowledge_bases:
        try:
            stats = await knowledge_base_service.get_knowledge_base_stats(kb)
            # Create a dict representation of the KB with stats
            kb_dict = {
                "id": kb.id,
                "name": kb.name,
                "description": kb.description,
                "chunk_size": kb.chunk_size,
                "chunk_overlap": kb.chunk_overlap,
                "embedding_model": kb.embedding_model,
                "settings": kb.settings,
                "is_active": kb.is_active,
                "workspace_id": kb.workspace_id,
                "collection_name": kb.collection_name,
                "user_id": kb.user_id,
                "created_at": kb.created_at,
                "updated_at": kb.updated_at,
                "full_collection_name": kb.full_collection_name,
                "document_count": stats.get("document_count", 0),
                "chunk_count": stats.get("chunk_count", 0)
            }
            enriched_kbs.append(kb_dict)
        except Exception as e:
            logger.warning(f"Failed to get stats for KB {kb.id}: {str(e)}")
            # Fallback to KB without stats
            kb_dict = {
                "id": kb.id,
                "name": kb.name,
                "description": kb.description,
                "chunk_size": kb.chunk_size,
                "chunk_overlap": kb.chunk_overlap,
                "embedding_model": kb.embedding_model,
                "settings": kb.settings,
                "is_active": kb.is_active,
                "workspace_id": kb.workspace_id,
                "collection_name": kb.collection_name,
                "user_id": kb.user_id,
                "created_at": kb.created_at,
                "updated_at": kb.updated_at,
                "full_collection_name": kb.full_collection_name,
                "document_count": None,
                "chunk_count": None
            }
            enriched_kbs.append(kb_dict)
    
    return enriched_kbs


@router.get("/{kb_id}", response_model=KnowledgeBasePublic)
async def get_knowledge_base(
    kb_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific knowledge base."""
    user_id: int = get_current_user_id(credentials)
    kb = await knowledge_base_service.get_knowledge_base_by_id(db, kb_id, user_id)
    
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    return kb


@router.get("/{kb_id}/stats")
async def get_knowledge_base_stats(
    kb_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """Get detailed stats for a knowledge base including Milvus collection info."""
    user_id: int = get_current_user_id(credentials)
    kb = await knowledge_base_service.get_knowledge_base_by_id(db, kb_id, user_id)
    
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Get stats from the service (includes Milvus collection stats)
    stats = await knowledge_base_service.get_knowledge_base_stats(kb)
    
    # Add knowledge base info
    result = {
        "id": kb.id,
        "name": kb.name,
        "description": kb.description,
        "collection_name": kb.collection_name,
        "full_collection_name": kb.full_collection_name,
        "is_active": kb.is_active,
        "created_at": kb.created_at,
        "updated_at": kb.updated_at,
        "stats": stats,
    }
    
    return result


@router.get("/{kb_id}/milvus-collection")
async def get_milvus_collection_details(
    kb_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """Get detailed Milvus collection information for debugging/verification."""
    user_id: int = get_current_user_id(credentials)
    kb = await knowledge_base_service.get_knowledge_base_by_id(db, kb_id, user_id)
    
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    try:
        # Get vector store instance
        vector_store = knowledge_base_service.get_vector_store(kb)
        
        # Get collection stats and info
        stats = vector_store.get_collection_stats()
        
        # Get all documents with their metadata
        documents = vector_store.list_all_documents()
        
        # Process documents to get unique sources and their chunk counts
        source_info = {}
        total_chunks = 0
        
        for text, metadata in documents:
            source_url = metadata.get('source_url', 'unknown')
            title = metadata.get('title', 'unknown')
            chunk_index = metadata.get('chunk_index', 0)
            total_doc_chunks = metadata.get('total_chunks', 0)
            content_type = metadata.get('content_type', 'unknown')
            
            if source_url not in source_info:
                source_info[source_url] = {
                    'title': title,
                    'content_type': content_type,
                    'chunks': [],
                    'total_chunks_expected': total_doc_chunks,
                    'chunk_count': 0
                }
            
            source_info[source_url]['chunks'].append({
                'chunk_index': chunk_index,
                'text_length': len(text),
                'metadata': metadata
            })
            source_info[source_url]['chunk_count'] += 1
            total_chunks += 1
        
        return {
            "knowledge_base": {
                "id": kb.id,
                "name": kb.name,
                "collection_name": kb.collection_name,
                "full_collection_name": kb.full_collection_name,
            },
            "milvus_stats": stats,
            "summary": {
                "unique_documents": len(source_info),
                "total_chunks": total_chunks,
                "total_entities_in_milvus": stats.get('total_entities', 0),
            },
            "documents": source_info,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error accessing Milvus collection: {str(e)}"
        )


@router.put("/{kb_id}", response_model=KnowledgeBasePublic)
async def update_knowledge_base(
    kb_id: int,
    request: KnowledgeBaseUpdate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """Update a knowledge base."""
    user_id: int = get_current_user_id(credentials)
    kb = await knowledge_base_service.update_knowledge_base(db, kb_id, request, user_id)
    
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    return kb


@router.delete("/{kb_id}")
async def delete_knowledge_base(
    kb_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """Delete a knowledge base and its associated Milvus collection."""
    user_id: int = get_current_user_id(credentials)
    success = await knowledge_base_service.delete_knowledge_base(db, kb_id, user_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    return {"message": f"Knowledge base {kb_id} deleted successfully"}
