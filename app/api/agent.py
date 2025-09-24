# TODO: update key for the knowledge base

"""
Agent Router for Knowledge Base Management with Authentication
Provides endpoints for knowledge base operations and agent communication
with JWT authentication and user-specific data isolation.
"""

import asyncio
import io
import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pymilvus import MilvusClient, utility
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.models.message import MessageRole
from app.schemas.conversation import ConversationCreate, MessageCreate
from app.schemas.knowledge_base import (
    KnowledgeBaseCreate,
    KnowledgeBasePublic,
    KnowledgeBaseWithStats,
)
from app.services.conversation_service import conversation_service, message_service
from app.services.knowledge_base_service import knowledge_base_service
from app.utils.auth import get_current_user_id
from engine.llm.agent import KnowledgeBaseAgent
from engine.rag.rag_system import RAGSystem
from engine.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


async def _get_knowledge_base_from_db(kb_name: str, user_id: int, db: AsyncSession):
    """
    Get a knowledge base from database by name for a specific user.
    
    Returns both the database record and VectorStore instance.
    """
    try:
        # Get knowledge base from database
        kb = await knowledge_base_service.get_knowledge_base_by_name(db, kb_name, user_id)
        if not kb:
            raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_name}' not found")
        
        # Get VectorStore instance
        vector_store = knowledge_base_service.get_vector_store(kb)
        
        return kb, vector_store
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting knowledge base {kb_name} for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get knowledge base: {str(e)}")


# Pydantic models for request/response
class CreateKnowledgeBaseRequest(BaseModel):
    name: str
    project_name: str
    description: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200


class KnowledgeBaseInfo(BaseModel):
    name: str
    description: Optional[str]
    document_count: int
    chunk_count: int
    created_at: str
    stats: Dict[str, Any]


class AddDocumentRequest(BaseModel):
    url: str


class AddTextRequest(BaseModel):
    text: str
    metadata: Dict[str, Any]


class QueryAgentRequest(BaseModel):
    message: str
    knowledge_base_name: str
    conversation_id: Optional[int] = None


class ChatAgentRequest(BaseModel):
    message: str
    knowledge_base_name: str
    conversation_id: Optional[int] = None


class AgentResponse(BaseModel):
    answer: str
    confidence: float
    processing_time: float
    sources: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class SelectiveSearchRequest(BaseModel):
    message: str
    knowledge_base_name: str
    document_titles: List[str]
    conversation_id: Optional[int] = None


class GenerateIdeaRequest(BaseModel):
    knowledge_base_name: str
    topic: Optional[str] = None
    content_type: Optional[str] = "article"  # article, blog_post, summary, etc.
    max_length: Optional[int] = 1000
    conversation_id: Optional[int] = None


class ConversationWithMessages(BaseModel):
    id: int
    conversation_name: str
    project_id: int
    knowledge_base_id: Optional[int]
    user_id: int
    created_at: str
    updated_at: str
    messages: List[Dict[str, Any]]


@router.post("/knowledge-bases", response_model=Dict[str, str])
async def create_knowledge_base(
    request: CreateKnowledgeBaseRequest,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Create a new knowledge base for the authenticated user using database service."""
    try:
        # Get workspace by project_name or throw error
        from app.services.workspace_service import WorkspaceService
        
        workspace_service = WorkspaceService()
        
        # Get workspace by name
        workspace = await workspace_service.get_workspace_by_name(db, request.project_name, current_user_id)
        
        if not workspace:
            raise HTTPException(
                status_code=404, 
                detail=f"Workspace '{request.project_name}' not found. Please create the workspace first."
            )

        name = f"chat_{current_user_id}_{int(time.time())}"
        
        # Create knowledge base data for service
        kb_create = KnowledgeBaseCreate(
            name=name,
            description=request.description,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            workspace_id=workspace.id
        )
        
        # Create knowledge base using service
        kb = await knowledge_base_service.create_knowledge_base(db, kb_create, current_user_id)
        
        logger.info(f"Created knowledge base: {request.name} for user {current_user_id}")
        
        return {
            "message": f"Knowledge base '{request.name}' created successfully",
            "name": request.name,
            "project_name": request.project_name,
            "collection_name": kb.full_collection_name,
            "user_id": str(current_user_id),
            "id": str(kb.id)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create knowledge base: {str(e)}")


@router.get("/knowledge-bases", response_model=List[KnowledgeBaseInfo])
async def list_knowledge_bases(
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """List all available knowledge bases for the authenticated user from database."""
    try:
        # Get knowledge bases from database
        knowledge_bases_db = await knowledge_base_service.list_user_knowledge_bases(db, current_user_id)
        
        knowledge_bases = []
        for kb in knowledge_bases_db:
            try:
                # Get statistics from Milvus
                stats = await knowledge_base_service.get_knowledge_base_stats(kb)
                
                kb_info = KnowledgeBaseInfo(
                    name=kb.name,
                    description=kb.description or f"Knowledge base: {kb.name}",
                    document_count=stats.get("document_count", 0),
                    chunk_count=stats.get("chunk_count", 0),
                    created_at=kb.created_at.isoformat(),
                    stats={
                        "id": kb.id,
                        "collection_name": kb.full_collection_name,
                        "chunk_size": kb.chunk_size,
                        "chunk_overlap": kb.chunk_overlap,
                        "embedding_model": kb.embedding_model,
                        "is_active": kb.is_active,
                        "workspace_id": kb.workspace_id,
                        # **stats.get("milvus_stats", {})
                    }
                )
                knowledge_bases.append(kb_info)
                
            except Exception as e:
                logger.warning(f"Error processing knowledge base {kb.name}: {str(e)}")
                # Still include the KB even if stats fail
                kb_info = KnowledgeBaseInfo(
                    name=kb.name,
                    description=kb.description or f"Knowledge base: {kb.name}",
                    document_count=0,
                    chunk_count=0,
                    created_at=kb.created_at.isoformat(),
                    stats={
                        "id": kb.id,
                        "collection_name": kb.full_collection_name,
                        "error": str(e)
                    }
                )
                knowledge_bases.append(kb_info)
        
        return knowledge_bases
        
    except Exception as e:
        logger.error(f"Error listing knowledge bases: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list knowledge bases: {str(e)}")


@router.get("/knowledge-bases/{kb_name}", response_model=KnowledgeBaseInfo)
async def get_knowledge_base(
    kb_name: str,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Get information about a specific knowledge base for the authenticated user."""
    try:
        kb, vector_store = await _get_knowledge_base_from_db(kb_name, current_user_id, db)
        
        # Get statistics from Milvus
        stats = await knowledge_base_service.get_knowledge_base_stats(kb)
        
        return KnowledgeBaseInfo(
            name=kb.name,
            description=kb.description or f"Knowledge base: {kb.name}",
            document_count=stats.get("document_count", 0),
            chunk_count=stats.get("chunk_count", 0),
            created_at=kb.created_at.isoformat(),
            stats={
                "id": kb.id,
                "collection_name": kb.full_collection_name,
                "chunk_size": kb.chunk_size,
                "chunk_overlap": kb.chunk_overlap,
                "embedding_model": kb.embedding_model,
                "is_active": kb.is_active,
                "workspace_id": kb.workspace_id,
                **stats.get("milvus_stats", {})
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting knowledge base {kb_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get knowledge base: {str(e)}")


@router.delete("/knowledge-bases/{kb_name}")
async def delete_knowledge_base(
    kb_name: str,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Delete a knowledge base and all its data using database service."""
    try:
        kb, _ = await _get_knowledge_base_from_db(kb_name, current_user_id, db)
        
        # Delete using service (handles both database and Milvus)
        success = await knowledge_base_service.delete_knowledge_base(db, kb.id, current_user_id)
        
        if success:
            logger.info(f"Deleted knowledge base: {kb_name} for user {current_user_id}")
            return {"message": f"Knowledge base '{kb_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_name}' not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting knowledge base {kb_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete knowledge base: {str(e)}")


@router.post("/knowledge-bases/{kb_name}/documents/url")
async def add_document_from_url(
    kb_name: str, 
    request: AddDocumentRequest,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Add a document to the knowledge base from a URL using VectorStore."""
    try:
        kb, vector_store = await _get_knowledge_base_from_db(kb_name, current_user_id, db)
        
        # Import required services for document processing
        from datetime import datetime

        from engine.rag.document_splitter import DocumentSplitter
        from engine.services.extractor import Extractor
        
        start_time = datetime.now()
        
        # Initialize extractor and splitter
        extractor = Extractor()
        splitter = DocumentSplitter()
        
        # Extract content from URL
        logger.info(f"Extracting content from: {request.url}")
        extracted_content = extractor.process_content(request.url)
        
        if not extracted_content.get("success", False):
            error_msg = extracted_content.get("error_message", "Unknown extraction error")
            logger.error(f"Extraction failed for {request.url}: {error_msg}")
            raise HTTPException(status_code=400, detail=f"Failed to extract content: {error_msg}")
        
        # Split content into chunks
        logger.info(f"Splitting content into chunks...")
        chunks = splitter.split_extracted_content(extracted_content)
        
        if not chunks:
            error_msg = "No content could be extracted for chunking"
            logger.warning(f"No chunks created for {request.url}: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Prepare texts and metadata for vector store
        texts = [chunk.text for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            metadata = chunk.metadata.copy()
            # Add user_id to metadata
            metadata['user_id'] = current_user_id
            metadatas.append(metadata)
        
        # Add to vector store
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        document_ids = vector_store.add_documents(texts, metadatas)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "message": "Document added successfully",
            "source_url": request.url,
            "status": "completed",
            "chunks_created": len(chunks),
            "processing_time": processing_time,
            "document_ids": document_ids
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding document from URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")


@router.post("/knowledge-bases/{kb_name}/documents/text")
async def add_text_document(
    kb_name: str, 
    request: AddTextRequest,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Add a text document to the knowledge base using VectorStore."""
    try:
        kb, vector_store = await _get_knowledge_base_from_db(kb_name, current_user_id, db)
        
        # Import required services for document processing
        from datetime import datetime

        from engine.rag.document_splitter import DocumentSplitter
        
        start_time = datetime.now()
        
        # Create metadata with proper structure
        source_id = request.metadata.get('source_url', f"custom_text_{kb_name}_{datetime.now().isoformat()}")
        metadata = {
            "source_url": source_id,
            "title": request.metadata.get('title', 'Custom Text'),
            "content_type": "text/plain",
            "extraction_time": datetime.now().isoformat(),
            "transcription_type": "custom",
            "user_id": current_user_id,
            **request.metadata
        }
        
        # Split text into chunks using DocumentSplitter
        splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_custom_text(request.text, metadata)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks could be created from the text")
        
        # Prepare texts and metadata for vector store
        texts = [chunk.text for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            chunk_metadata = chunk.metadata.copy()
            # Ensure user_id is in metadata
            chunk_metadata['user_id'] = current_user_id
            # Add project_name if available in request metadata
            if 'project_name' in request.metadata:
                chunk_metadata['project_name'] = request.metadata['project_name']
            metadatas.append(chunk_metadata)
        
        # Add to vector store
        document_ids = vector_store.add_documents(texts, metadatas)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "message": "Text document added successfully",
            "source_url": source_id,
            "status": "completed",
            "chunks_created": len(chunks),
            "processing_time": processing_time,
            "document_ids": document_ids
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding text document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add text document: {str(e)}")


@router.post("/knowledge-bases/{kb_name}/documents/file")
async def add_document_from_file(
    kb_name: str, 
    file: UploadFile = File(...),
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Add a document to the knowledge base from an uploaded file using Extractor."""
    try:
        kb, vector_store = await _get_knowledge_base_from_db(kb_name, current_user_id, db)
        
        # Import required services for document processing
        from datetime import datetime

        from engine.rag.document_splitter import DocumentSplitter
        from engine.services.extractor import Extractor
        
        start_time = datetime.now()
        
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.txt', '.doc'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Create temporary file to save uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Read and write file content
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Initialize extractor and splitter
            extractor = Extractor()
            splitter = DocumentSplitter()
            
            # Process the file using extractor
            logger.info(f"Processing uploaded file: {file.filename}")
            extracted_content = extractor.process_content(temp_file_path)
            
            if not extracted_content.get("success", False):
                error_msg = extracted_content.get("error_message", "Unknown extraction error")
                logger.error(f"Extraction failed for {file.filename}: {error_msg}")
                raise HTTPException(status_code=400, detail=f"Failed to extract content: {error_msg}")
            
            # Override the source_url to use the original filename instead of temp path
            extracted_content["url"] = file.filename
            if "metadata" not in extracted_content:
                extracted_content["metadata"] = {}
            extracted_content["metadata"]["title"] = file.filename
            
            # Split content into chunks
            logger.info(f"Splitting content into chunks...")
            chunks = splitter.split_extracted_content(extracted_content)
            
            if not chunks:
                error_msg = "No content could be extracted for chunking"
                logger.warning(f"No chunks created for {file.filename}: {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Prepare texts and metadata for vector store
            texts = [chunk.text for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                metadata = chunk.metadata.copy()
                # Add user_id and original filename to metadata
                metadata['user_id'] = current_user_id
                metadata['original_filename'] = file.filename
                metadatas.append(metadata)
            
            # Add to vector store
            logger.info(f"Adding {len(chunks)} chunks to vector store...")
            document_ids = vector_store.add_documents(texts, metadatas)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "message": "File document added successfully",
                "filename": file.filename,
                "status": "completed",
                "chunks_created": len(chunks),
                "processing_time": processing_time,
                "document_ids": document_ids
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding file document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add file document: {str(e)}")


@router.post("/knowledge-bases/{kb_name}/documents/audio")
async def add_audio_file(
    kb_name: str, 
    file: UploadFile = File(...),
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Add an audio file to the knowledge base by transcribing it using Extractor."""
    try:
        kb, vector_store = await _get_knowledge_base_from_db(kb_name, current_user_id, db)
        
        # Import required services for audio processing
        from datetime import datetime
        
        start_time = datetime.now()
        
        # Validate file type
        allowed_extensions = {'.mp3', '.wav', '.m4a', '.mp4', '.avi', '.mov', '.flv', '.webm'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported audio file type: {file_extension}. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Create temporary file to save uploaded audio content
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Read and write file content
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Initialize RAG system with the sanitized collection name
            collection_name = kb_name.lower().replace(' ', '_')
            rag_system = RAGSystem(collection_name=collection_name)
            
            # Process the audio file using RAG system (it handles transcription via extractor)
            logger.info(f"Processing uploaded audio file: {file.filename}")
            
            # Use RAG system to process the audio file directly
            # The extractor within RAG system will handle transcription
            status = rag_system.add_document_from_url(temp_file_path)
            
            if status.status != "completed":
                error_msg = status.error_message or "Unknown transcription error"
                logger.error(f"Transcription failed for {file.filename}: {error_msg}")
                raise HTTPException(status_code=400, detail=f"Failed to transcribe audio: {error_msg}")
            
            # Update the source_url in the vector store to use the original filename
            # First delete the document with temp path, then re-add with correct source_url
            rag_system.delete_document(temp_file_path)
            
            # Re-process with the correct source URL by temporarily renaming the file
            import shutil
            correct_path = os.path.join(os.path.dirname(temp_file_path), file.filename)
            shutil.copy2(temp_file_path, correct_path)
            
            try:
                status = rag_system.add_document_from_url(correct_path)
                if status.status != "completed":
                    error_msg = status.error_message or "Unknown transcription error"
                    logger.error(f"Re-processing failed for {file.filename}: {error_msg}")
                    raise HTTPException(status_code=400, detail=f"Failed to transcribe audio: {error_msg}")
            finally:
                # Clean up the copied file
                if os.path.exists(correct_path):
                    os.unlink(correct_path)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "message": "Audio file transcribed and added successfully",
                "filename": file.filename,
                "file_type": file_extension,
                "status": "completed",
                "chunks_created": status.chunks_created,
                "processing_time": processing_time,
                "source_url": status.source_url,
                "transcription_info": {
                    "original_audio_file": file.filename,
                    "transcription_method": "deepgram_api"
                }
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding audio file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add audio file: {str(e)}")


@router.get("/knowledge-bases/{kb_name}/documents")
async def list_documents(
    kb_name: str,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """List all documents in the knowledge base for the authenticated user."""
    try:
        kb, vector_store = await _get_knowledge_base_from_db(kb_name, current_user_id, db)
        
        # Get all documents from vector store
        documents = vector_store.list_all_documents()
        
        # Group by source URL to create document summaries
        doc_map = {}
        for text, metadata in documents:
            source_url = metadata.get("source_url", "unknown")
            if source_url not in doc_map:
                # Use original_filename if available, otherwise fall back to title or source_url
                display_name = metadata.get("original_filename") or metadata.get("title", "") or source_url
                doc_map[source_url] = {
                    "source_url": source_url,
                    "content_type": metadata.get("content_type", ""),
                    "title": display_name,
                    "extraction_time": metadata.get("extraction_time", ""),
                    "total_chunks": 0,
                    "total_characters": 0
                }
            
            doc_map[source_url]["total_chunks"] += 1
            doc_map[source_url]["total_characters"] += len(text)
        
        documents_list = list(doc_map.values())
        
        return {
            "knowledge_base": kb_name,
            "document_count": len(documents_list),
            "documents": documents_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.delete("/knowledge-bases/{kb_name}/documents")
async def delete_document(
    kb_name: str, 
    source_url: str,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Delete a specific document from the knowledge base for the authenticated user."""
    try:
        kb, vector_store = await _get_knowledge_base_from_db(kb_name, current_user_id, db)
        
        # Delete document by source URL
        deleted_count = vector_store.delete_by_source_url(source_url)
        
        if deleted_count > 0:
            return {
                "message": f"Document '{source_url}' deleted successfully",
                "chunks_deleted": deleted_count
            }
        else:
            raise HTTPException(status_code=404, detail=f"Document '{source_url}' not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.post("/knowledge-bases/{kb_name}/query")
async def query_agent(
    kb_name: str, 
    request: QueryAgentRequest,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Query the agent using the specified knowledge base and return streaming response."""
    try:
        # Get the knowledge base from database
        kb, vector_store = await _get_knowledge_base_from_db(kb_name, current_user_id, db)
        
        # Create agent for the knowledge base on demand
        collection_name = vector_store.collection_name
        agent = KnowledgeBaseAgent(rag_collection_name=collection_name)
        
        # Create streaming response generator
        async def generate_response():
            try:
                # Get streaming response from agent
                for chunk in agent.process_query_stream(request.message, []):
                    yield f"data: {chunk}\n\n"
                
                # Send end of stream marker
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming response: {str(e)}")
                yield f"data: Error: {str(e)}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to query agent: {str(e)}")


@router.get("/conversations/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation_with_messages(
    conversation_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Get a conversation with all its messages for the authenticated user."""
    try:
        # Get conversation with messages
        conversation = await conversation_service.get_conversation_by_id(db, conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Verify the conversation belongs to the current user
        if conversation.user_id != current_user_id:
            raise HTTPException(status_code=403, detail="Access denied to this conversation")
        
        # Format messages
        messages = []
        if conversation.messages:
            for msg in conversation.messages:
                messages.append({
                    "id": msg.id,
                    "content": msg.content,
                    "role": msg.role.value,
                    "created_at": msg.created_at.isoformat(),
                    "user_id": msg.user_id
                })
        
        return ConversationWithMessages(
            id=conversation.id,
            conversation_name=conversation.conversation_name,
            project_id=conversation.project_id,
            knowledge_base_id=conversation.knowledge_base_id,
            user_id=conversation.user_id,
            created_at=conversation.created_at.isoformat(),
            updated_at=conversation.updated_at.isoformat(),
            messages=messages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation: {str(e)}")


@router.post("/chat-agent")
async def chat_agent(
    request: ChatAgentRequest,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Chat with the agent using streaming response.
    Creates a new conversation if conversation_id is not provided.
    Returns only the agent's answer without thinking steps.
    """
    try:
        conversation_id = request.conversation_id
        
        # If no conversation_id provided, create a new conversation
        if conversation_id is None:
            # Get the knowledge base first to link it to the conversation
            kb, _ = await _get_knowledge_base_from_db(request.knowledge_base_name, current_user_id, db)
            
            # Use the knowledge base's workspace
            if not kb.workspace_id:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Knowledge base '{request.knowledge_base_name}' is not associated with any workspace"
                )
            
            project_id = kb.workspace_id
            
            conversation_data = ConversationCreate(
                conversation_name=f"Chat with {request.knowledge_base_name}",
                project_id=project_id,
                knowledge_base_id=kb.id  # Link conversation to knowledge base
            )
            conversation = await conversation_service.create_conversation(
                db=db,
                conversation_data=conversation_data,
                user_id=current_user_id
            )
            conversation_id = conversation.id
        else:
            # Verify the conversation belongs to the current user
            conversation = await conversation_service.get_conversation_by_id(db, conversation_id)
            if not conversation or conversation.user_id != current_user_id:
                raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Save user message to database
        user_message_data = MessageCreate(
            content=request.message,
            role=MessageRole.user,
            conversation_id=conversation_id,
            user_id=current_user_id
        )
        user_message = await message_service.create_message(
            db=db,
            message_data=user_message_data
        )
        
        # Load conversation history (last 10 messages for context)
        conversation_with_messages = await conversation_service.get_conversation_by_id(db, conversation_id)
        conversation_history = []
        if conversation_with_messages and conversation_with_messages.messages:
            # Get last 10 messages, excluding the current user message
            recent_messages = conversation_with_messages.messages[-11:-1] if len(conversation_with_messages.messages) > 1 else []
            for msg in recent_messages:
                conversation_history.append({
                    "role": msg.role.value.lower(),
                    "content": msg.content
                })
        
        # Get the knowledge base from database
        kb, vector_store = await _get_knowledge_base_from_db(request.knowledge_base_name, current_user_id, db)
        
        # Initialize the agent
        collection_name = vector_store.collection_name
        agent = KnowledgeBaseAgent(rag_collection_name=collection_name)
        
        async def generate_response():
            """Generate streaming response with only the agent's answer"""
            try:
                full_response = ""
                
                # Stream the response from the agent
                for chunk in agent.process_query_stream(request.message, conversation_history):
                    if chunk:
                        full_response += chunk
                        # Send only the content without any thinking steps or metadata
                        yield f"data: {chunk}\n\n"
                        # Add 0.1 second delay for streaming effect
                        await asyncio.sleep(0.1)
                        # Add 0.1 second delay for streaming effect
                        await asyncio.sleep(0.1)
                
                # Save agent response to database
                agent_message_data = MessageCreate(
                    content=full_response,
                    role=MessageRole.agent,
                    conversation_id=conversation_id,
                    user_id=current_user_id
                )
                await message_service.create_message(
                    db=db,
                    message_data=agent_message_data
                )
                
                # Send conversation_id in the final message for client reference
                yield f"data: [CONVERSATION_ID:{conversation_id}]\n\n"
                yield f"data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming response: {str(e)}")
                yield f"data: [ERROR: {str(e)}]\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat: {str(e)}")


@router.post("/knowledge-bases/{kb_name}/selective-search")
async def selective_search_agent(
    kb_name: str,
    request: SelectiveSearchRequest,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Chat with the agent using streaming response, searching only within specific documents.
    Creates a new conversation if conversation_id is not provided.
    Returns only the agent's answer without thinking steps.
    """
    try:
        conversation_id = request.conversation_id
        
        # If no conversation_id provided, create a new conversation
        if conversation_id is None:
            # Get the knowledge base first to link it to the conversation
            kb, _ = await _get_knowledge_base_from_db(kb_name, current_user_id, db)
            
            # Use the knowledge base's workspace
            if not kb.workspace_id:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Knowledge base '{kb_name}' is not associated with any workspace"
                )
            
            project_id = kb.workspace_id
            
            conversation_data = ConversationCreate(
                conversation_name=f"Selective Search in {kb_name}",
                project_id=project_id,
                knowledge_base_id=kb.id  # Link conversation to knowledge base
            )
            conversation = await conversation_service.create_conversation(
                db=db,
                conversation_data=conversation_data,
                user_id=current_user_id
            )
            conversation_id = conversation.id
        else:
            # Verify the conversation belongs to the current user
            conversation = await conversation_service.get_conversation_by_id(db, conversation_id)
            if not conversation or conversation.user_id != current_user_id:
                raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Save user message to database
        user_message_data = MessageCreate(
            content=request.message,
            role=MessageRole.user,
            conversation_id=conversation_id,
            user_id=current_user_id
        )
        user_message = await message_service.create_message(
            db=db,
            message_data=user_message_data
        )
        
        # Load conversation history (last 10 messages for context)
        conversation_with_messages = await conversation_service.get_conversation_by_id(db, conversation_id)
        conversation_history = []
        if conversation_with_messages and conversation_with_messages.messages:
            # Get last 10 messages, excluding the current user message
            recent_messages = conversation_with_messages.messages[-11:-1] if len(conversation_with_messages.messages) > 1 else []
            for msg in recent_messages:
                # Convert database role to LLM API role
                api_role = "assistant" if msg.role.value == "agent" else msg.role.value
                conversation_history.append({
                    "role": api_role,
                    "content": msg.content
                })
        
        # Get the knowledge base from database
        kb, vector_store = await _get_knowledge_base_from_db(kb_name, current_user_id, db)
        
        # Filter documents by the provided titles
        all_documents = vector_store.list_all_documents()
        matching_source_urls = set()
        
        for _, metadata in all_documents:
            doc_title = metadata.get('title', '').strip()
            original_filename = metadata.get('original_filename', '').strip()
            
            # Check if any of the requested titles match this document
            for requested_title in request.document_titles:
                requested_title = requested_title.strip()
                
                # Match against title or original filename (case-insensitive)
                if (doc_title and requested_title.lower() in doc_title.lower()) or \
                   (original_filename and requested_title.lower() in original_filename.lower()) or \
                   (requested_title.lower() == doc_title.lower()) or \
                   (requested_title.lower() == original_filename.lower()):
                    matching_source_urls.add(metadata.get('source_url', ''))
                    break
        
        # Create a custom agent that searches only within selected documents
        class SelectiveKnowledgeBaseAgent:
            def __init__(self, vector_store, selected_source_urls):
                self.vector_store = vector_store
                self.selected_source_urls = selected_source_urls
                # Initialize the base agent for LLM functionality
                collection_name = vector_store.collection_name
                self.base_agent = KnowledgeBaseAgent(rag_collection_name=collection_name)
            
            async def process_query_stream(self, query, conversation_history=None):
                """Process query with selective document filtering"""
                if not self.selected_source_urls:
                    yield "No documents found matching the specified titles."
                    return
                
                # Convert set to list for the vector store method
                source_urls_list = list(self.selected_source_urls)
                if not source_urls_list or not any(url for url in source_urls_list):
                    yield "No valid source URLs found for filtering."
                    return
                
                # Perform filtered similarity search
                try:
                    search_results = self.vector_store.similarity_search(
                        query=query,
                        k=5,
                        source_urls=source_urls_list
                    )
                    
                    if not search_results:
                        yield f"No relevant content found in the specified documents for the query: '{query}'"
                        return
                    
                    # Create context from search results (search_results is a list of tuples: (text, distance, metadata))
                    context_texts = [result[0] for result in search_results]  # Extract text from tuple
                    context = "\n\n".join(context_texts)
                    
                    # Generate response using the base agent's LLM with filtered context
                    try:
                        # Create a prompt with the filtered context
                        prompt = f"""Based on the following context from the selected documents, answer the user's question.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based only on the information provided in the context above."""
                        
                        # Use the base agent's LLM to generate response
                        llm_client = self.base_agent.llm_client
                        
                        # Prepare conversation history in the correct format
                        messages = []
                        if conversation_history:
                            for msg in conversation_history:
                                messages.append({
                                    "role": msg["role"],
                                    "content": msg["content"]
                                })
                        
                        # Add the current prompt
                        messages.append({
                            "role": "user",
                            "content": prompt
                        })
                        
                        # Stream response from LLM
                        for chunk in llm_client.chat_stream(messages):
                            if chunk:
                                yield chunk
                                
                    except Exception as llm_error:
                        logger.error(f"Error generating LLM response: {str(llm_error)}")
                        yield f"Error generating response: {str(llm_error)}"
                        
                except Exception as e:
                    logger.error(f"Error in selective search: {str(e)}")
                    yield f"Error performing selective search: {str(e)}"
        
        # Initialize the selective agent
        agent = SelectiveKnowledgeBaseAgent(vector_store, matching_source_urls)
        
        async def generate_response():
            """Generate streaming response with only the agent's answer"""
            try:
                full_response = ""
                
                # Stream the response from the agent
                for chunk in agent.process_query_stream(request.message, conversation_history):
                    if chunk:
                        full_response += chunk
                        # Send only the content without any thinking steps or metadata
                        yield f"data: {chunk}\n\n"
                
                # Save agent response to database
                agent_message_data = MessageCreate(
                    content=full_response,
                    role=MessageRole.agent,
                    conversation_id=conversation_id,
                    user_id=current_user_id
                )
                await message_service.create_message(
                    db=db,
                    message_data=agent_message_data
                )
                
                # Send conversation_id in the final message for client reference
                yield f"data: [CONVERSATION_ID:{conversation_id}]\n\n"
                yield f"data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming response: {str(e)}")
                yield f"data: [ERROR: {str(e)}]\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in selective search agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process selective search: {str(e)}")


@router.post("/knowledge-bases/{kb_name}/generate-idea")
async def generate_idea(
    kb_name: str,
    request: GenerateIdeaRequest,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate new content ideas based on all documents in the knowledge base.
    Creates a new conversation if conversation_id is not provided.
    Returns streaming response with generated content.
    """
    try:
        conversation_id = request.conversation_id
        
        # If no conversation_id provided, create a new conversation
        if conversation_id is None:
            # Get the knowledge base first to link it to the conversation
            kb, _ = await _get_knowledge_base_from_db(kb_name, current_user_id, db)
            
            # Use the knowledge base's workspace
            if not kb.workspace_id:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Knowledge base '{kb_name}' is not associated with any workspace"
                )
            
            project_id = kb.workspace_id
            
            conversation_data = ConversationCreate(
                conversation_name=f"Idea Generation from {kb_name}",
                project_id=project_id,
                knowledge_base_id=kb.id  # Link conversation to knowledge base
            )
            conversation = await conversation_service.create_conversation(
                db=db,
                conversation_data=conversation_data,
                user_id=current_user_id
            )
            conversation_id = conversation.id
        else:
            # Verify the conversation belongs to the current user
            conversation = await conversation_service.get_conversation_by_id(db, conversation_id)
            if not conversation or conversation.user_id != current_user_id:
                raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get the knowledge base from database
        kb, vector_store = await _get_knowledge_base_from_db(kb_name, current_user_id, db)
        
        # Retrieve all documents from the knowledge base
        all_documents = vector_store.list_all_documents()
        
        if not all_documents:
            raise HTTPException(
                status_code=404, 
                detail=f"No documents found in knowledge base '{kb_name}'"
            )
        
        # Prepare content for idea generation
        document_summaries = []
        unique_sources = set()
        
        for text, metadata in all_documents:
            source_url = metadata.get('source_url', '')
            title = metadata.get('title', 'Untitled')
            content_type = metadata.get('content_type', 'unknown')
            
            # Avoid duplicate sources and create summaries
            if source_url not in unique_sources:
                unique_sources.add(source_url)
                # Take first 500 characters as summary
                summary = text[:500] + "..." if len(text) > 500 else text
                document_summaries.append({
                    'title': title,
                    'content_type': content_type,
                    'summary': summary,
                    'source_url': source_url
                })
        
        # Create the idea generation prompt
        topic_context = f" focusing on the topic: {request.topic}" if request.topic else ""
        content_type_instruction = f"Format the output as a {request.content_type}."
        
        idea_prompt = f"""Based on the following documents from the knowledge base, generate new creative content ideas{topic_context}.

Available Documents:
"""
        
        for i, doc in enumerate(document_summaries[:20], 1):  # Limit to 20 documents to avoid token limits
            idea_prompt += f"\n{i}. Title: {doc['title']}\n   Type: {doc['content_type']}\n   Summary: {doc['summary']}\n"
        
        idea_prompt += f"""

Task: Generate innovative and creative content based on the themes, concepts, and information from these documents. {content_type_instruction}

Requirements:
- Create original content that synthesizes information from multiple sources
- Identify patterns, connections, and insights across the documents
- Suggest new perspectives or applications of the existing knowledge
- Keep the response under {request.max_length} words
- Be creative and think outside the box while staying grounded in the source material

Generate your response now:"""
        
        # Save the generation request as a user message
        user_message_data = MessageCreate(
            content=f"Generate idea for {request.content_type}" + (f" about {request.topic}" if request.topic else ""),
            role=MessageRole.USER,
            conversation_id=conversation_id,
            user_id=current_user_id
        )
        user_message = await message_service.create_message(
            db=db,
            message_data=user_message_data
        )
        
        # Initialize the agent for idea generation
        agent = KnowledgeBaseAgent(vector_store)
        
        async def generate_response():
            """Generate streaming response with the generated idea"""
            try:
                full_response = ""
                
                # Use the agent's LLM client directly for idea generation
                async for chunk in agent.llm_client.stream_completion(
                    messages=[{"role": "user", "content": idea_prompt}],
                    max_tokens=request.max_length,
                    temperature=0.7  # Higher temperature for more creativity
                ):
                    if chunk:
                        full_response += chunk
                        yield f"data: {chunk}\n\n"
                        # Add 0.1 second delay for streaming effect
                        await asyncio.sleep(0.1)
                
                # Save agent response to database
                agent_message_data = MessageCreate(
                    content=full_response,
                    role=MessageRole.AGENT,
                    conversation_id=conversation_id,
                    user_id=current_user_id
                )
                await message_service.create_message(
                    db=db,
                    message_data=agent_message_data
                )
                
                # Send conversation_id in the final message for client reference
                yield f"data: [CONVERSATION_ID:{conversation_id}]\n\n"
                yield f"data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Error in idea generation streaming response: {str(e)}")
                yield f"data: [ERROR: {str(e)}]\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generate idea: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate idea: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for the test router with Milvus status."""
    try:
        # Get basic router status
        router_status = {
            "status": "healthy",
            "message": "Test Agent Router is running"
        }
        
        # Get Milvus status using VectorStore
        milvus_status = {}
        try:
            # Create a temporary VectorStore instance to check Milvus connection
            temp_vector_store = VectorStore(collection_name="health_check_temp")
            
            # Connect directly to Milvus for more detailed status
            milvus_host = settings.MILVUS_HOST
            milvus_port = settings.MILVUS_PORT
            milvus_uri = f"http://{milvus_host}:{milvus_port}"
            
            client = MilvusClient(uri=milvus_uri)
            
            # Get Milvus server info and collections
            collections = client.list_collections()
            
            # Count test collections
            test_collections = [c for c in collections if c.startswith("test_kb_")]
            
            # Get total statistics across all test collections
            total_chunks = 0
            collection_stats = {}
            
            for collection_name in test_collections:
                try:
                    stats = client.get_collection_stats(collection_name)
                    row_count = stats.get("row_count", 0)
                    total_chunks += row_count
                    collection_stats[collection_name] = {
                        "row_count": row_count,
                        "status": "active"
                    }
                except Exception as e:
                    collection_stats[collection_name] = {
                        "row_count": 0,
                        "status": "error",
                        "error": str(e)
                    }
            
            milvus_status = {
                "connection": "healthy",
                "host": milvus_host,
                "port": milvus_port,
                "uri": milvus_uri,
                "total_collections": len(collections),
                "test_collections": len(test_collections),
                "total_test_chunks": total_chunks,
                "collections": collection_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting Milvus status: {str(e)}")
            milvus_status = {
                "connection": "error",
                "error": str(e),
                "host": settings.MILVUS_HOST,
                "port": settings.MILVUS_PORT
            }
        
        # Combine router and Milvus status
        return {
            **router_status,
            "milvus": milvus_status
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return {
            "status": "error",
            "message": "Health check failed",
            "error": str(e),
            "milvus": {
                "connection": "unknown",
                "error": "Failed to check Milvus status"
            }
        }
