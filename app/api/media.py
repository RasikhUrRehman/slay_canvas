"""
Media API router for file upload and document management.
Handles media file uploads and RAG document management capabilities.
"""

import os
import logging
import tempfile
import shutil
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.schemas.media import MediaCreate, MediaRead
from app.db.session import get_db
from app.services.media_service import create_media, MediaService
from app.services.auth_service import get_current_user
from app.models.user import User
from app.utils.auth import get_current_user_id

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from engine.rag_system import RAGSystem
    RAG_AVAILABLE = True
except ImportError:
    # Fallback if engine is not available
    RAGSystem = None
    RAG_AVAILABLE = False

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/media", tags=["media"])
security = HTTPBearer()

# Global RAG system instance
rag_system = None


# Pydantic models for document management
class URLRequest(BaseModel):
    url: str = Field(..., description="URL to extract and add to RAG system")
    title: Optional[str] = Field(default=None, description="Optional title for the document")


class TextRequest(BaseModel):
    text: str = Field(..., description="Text content to add")
    title: str = Field(..., description="Title for the text document")
    content_type: Optional[str] = Field(default="custom_text", description="Content type")
    source_url: Optional[str] = Field(default=None, description="Optional source URL")


class DocumentResponse(BaseModel):
    status: str
    message: str
    source_url: str
    chunks_created: int
    processing_time: float
    error_message: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[Dict[str, Any]]
    total_documents: int


class SystemStatsResponse(BaseModel):
    vector_store: Dict[str, Any]
    processing_status: Dict[str, int]
    total_processed_documents: int
    system_components: Dict[str, str]


def get_rag_system():
    """Get or initialize RAG system."""
    global rag_system
    
    if not RAG_AVAILABLE:
        logger.warning("RAG system not available - engine module not found")
        return None
        
    if rag_system is None:
        try:
            rag_system = RAGSystem(
                collection_name="media_documents",
                chunk_size=1000,
                chunk_overlap=200,
                top_k=5
            )
            logger.info("RAG system initialized for media")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return None
    return rag_system


async def process_document_background(rag, source: str, doc_type: str, user_id: int, **kwargs):
    """Background task for processing documents."""
    try:
        if doc_type == "url":
            status = rag.add_document_from_url(source)
        elif doc_type == "text":
            status = rag.add_custom_text(source, kwargs.get("metadata", {}))
        elif doc_type == "file":
            # For file processing, we'd implement file handling here
            # This is a placeholder for file processing logic
            pass
        
        logger.info(f"Background processing completed for {source} (user {user_id}): {status.status}")
        
    except Exception as e:
        logger.error(f"Background processing failed for {source} (user {user_id}): {e}")


# Traditional media upload endpoints
@router.post("/upload", response_model=MediaRead)
async def upload_media(
    file: UploadFile = File(...),
    title: str = Form(...),
    description: Optional[str] = Form(None),
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Upload a media file to the traditional media system."""
    try:
        # Create media record
        media_data = MediaCreate(
            title=title,
            description=description,
            file_path=f"uploads/{file.filename}",
            file_type=file.content_type,
            file_size=file.size
        )
        
        media = await create_media(db, media_data, user_id)
        return media
        
    except Exception as e:
        logger.error(f"Media upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-audio", response_model=MediaRead)
async def upload_audio(
    file: UploadFile = File(...),
    title: str = Form(...),
    description: Optional[str] = Form(None),
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload an audio file to the system.
    Supports all audio formats including MP3, WAV, FLAC, AAC, OGG, M4A, MP4, WMA, AIFF, and more.
    """
    try:
        # Define supported audio MIME types and file extensions
        supported_audio_types = {
            # Common audio formats
            'audio/mpeg',           # MP3
            'audio/wav',            # WAV
            #'audio/wave',           # WAV (alternative)
            #'audio/x-wav',          # WAV (alternative)
            # 'audio/flac',           # FLAC
            # 'audio/x-flac',         # FLAC (alternative)
            # 'audio/aac',            # AAC
            'audio/mp4',            # MP4 audio
            # 'audio/x-m4a',          # M4A
            # 'audio/m4a',            # M4A (alternative)
            # 'audio/ogg',            # OGG
            # 'audio/vorbis',         # OGG Vorbis
            # 'audio/x-ms-wma',       # WMA
            # 'audio/aiff',           # AIFF
            # 'audio/x-aiff',         # AIFF (alternative)
            # 'audio/amr',            # AMR
            # 'audio/3gpp',           # 3GP audio
            'audio/webm',           # WebM audio
            #'audio/opus',           # Opus
            
            # Video formats that contain audio (MP4, etc.)
            'video/mp4',            # MP4 video (contains audio)
            # 'video/quicktime',      # MOV (contains audio)
            # 'video/x-msvideo',      # AVI (contains audio)
            'video/webm',           # WebM video (contains audio)
        }
        
        supported_extensions = {
            '.mp3', '.wav', '.mp4', '.mov', 
            '.avi', '.webm', '.m4v', '.mkv'
        }
        
        # Get file extension
        file_extension = os.path.splitext(file.filename.lower())[1] if file.filename else ''
        
        # Validate audio file type
        is_valid_mime = file.content_type in supported_audio_types
        is_valid_extension = file_extension in supported_extensions
        
        if not (is_valid_mime or is_valid_extension):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported audio format. Supported formats include: MP3, WAV, FLAC, AAC, M4A, OGG, MP4, WMA, AIFF, and more. "
                       f"Received: {file.content_type} with extension {file_extension}"
            )
        
        # Validate file size (limit to 500MB for audio files)
        max_size = 500 * 1024 * 1024  # 500MB in bytes
        if file.size and file.size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Audio file too large. Maximum size is 500MB. Received: {file.size / (1024*1024):.1f}MB"
            )
        
        # Create uploads directory if it doesn't exist
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"audio_{timestamp}_{file.filename}"
        file_path = os.path.join(upload_dir, safe_filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create media record
        media_data = MediaCreate(
            title=title,
            description=description,
            file_path=file_path,
            file_type=file.content_type,
            file_size=file.size
        )
        
        media = await create_media(db, media_data, user_id)
        
        logger.info(f"Audio file uploaded successfully: {safe_filename} by user {user_id}")
        return media
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Audio upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload audio file: {str(e)}")


# RAG document management endpoints
@router.post("/upload-document", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
    content_type: Optional[str] = Form(default="uploaded_file"),
    process_async: Optional[bool] = Form(default=False),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload and process a document for RAG system.
    Requires authentication.
    """
    try:
        # Verify user authentication
        user = await get_current_user(db, credentials.credentials)
        
        rag = get_rag_system()
        if rag is None:
            raise HTTPException(status_code=500, detail="RAG system not available")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Create metadata
            metadata = {
                "source_url": f"file://{file.filename}",
                "content_type": content_type,
                "title": title or file.filename,
                "extraction_time": datetime.now().isoformat(),
                "original_filename": file.filename,
                "user_id": user.id  # Add user context
            }
            
            if process_async:
                # Process in background
                background_tasks.add_task(
                    process_document_background,
                    rag,
                    tmp_path,
                    "file",
                    user.id,
                    metadata=metadata
                )
                
                return DocumentResponse(
                    status="processing",
                    message="File uploaded and queued for processing",
                    source_url=metadata["source_url"],
                    chunks_created=0,
                    processing_time=0.0
                )
            else:
                # Read file content (simplified - in production, use proper file processors)
                with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                status = rag.add_custom_text(content, metadata)
                
                return DocumentResponse(
                    status=status.status,
                    message="File processed successfully" if status.status == "completed" else "File processing failed",
                    source_url=status.source_url,
                    chunks_created=status.chunks_created,
                    processing_time=status.processing_time,
                    error_message=status.error_message
                )
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add-url", response_model=DocumentResponse)
async def add_url(
    request: URLRequest,
    background_tasks: BackgroundTasks,
    process_async: Optional[bool] = False,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    Add a document from URL to RAG system.
    Requires authentication.
    """
    try:
        # Verify user authentication
        user = await get_current_user(db, credentials.credentials)
        
        rag = get_rag_system()
        if rag is None:
            raise HTTPException(status_code=500, detail="RAG system not available")
        
        if process_async:
            # Process in background
            background_tasks.add_task(
                process_document_background,
                rag,
                request.url,
                "url",
                user.id
            )
            
            return DocumentResponse(
                status="processing",
                message="URL queued for processing",
                source_url=request.url,
                chunks_created=0,
                processing_time=0.0
            )
        else:
            status = rag.add_document_from_url(request.url)
            
            return DocumentResponse(
                status=status.status,
                message="URL processed successfully" if status.status == "completed" else "URL processing failed",
                source_url=status.source_url,
                chunks_created=status.chunks_created,
                processing_time=status.processing_time,
                error_message=status.error_message
            )
            
    except Exception as e:
        logger.error(f"URL processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add-text", response_model=DocumentResponse)
async def add_text(
    request: TextRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    Add custom text content to RAG system.
    Requires authentication.
    """
    try:
        # Verify user authentication
        user = await get_current_user(db, credentials.credentials)
        
        rag = get_rag_system()
        if rag is None:
            raise HTTPException(status_code=500, detail="RAG system not available")
        
        metadata = {
            "source_url": request.source_url or f"text_document_{datetime.now().timestamp()}",
            "content_type": request.content_type,
            "title": request.title,
            "extraction_time": datetime.now().isoformat(),
            "user_id": user.id  # Add user context
        }
        
        status = rag.add_custom_text(request.text, metadata)
        
        return DocumentResponse(
            status=status.status,
            message="Text processed successfully" if status.status == "completed" else "Text processing failed",
            source_url=status.source_url,
            chunks_created=status.chunks_created,
            processing_time=status.processing_time,
            error_message=status.error_message
        )
        
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    List all documents in the RAG system.
    Requires authentication.
    """
    try:
        # Verify user authentication
        user = await get_current_user(db, credentials.credentials)
        
        rag = get_rag_system()
        if rag is None:
            raise HTTPException(status_code=500, detail="RAG system not available")
        
        documents = rag.list_documents()
        
        return DocumentListResponse(
            documents=documents,
            total_documents=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{source_url:path}")
async def delete_document(
    source_url: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a document by source URL from RAG system.
    Requires authentication.
    """
    try:
        # Verify user authentication
        user = await get_current_user(db, credentials.credentials)
        
        rag = get_rag_system()
        if rag is None:
            raise HTTPException(status_code=500, detail="RAG system not available")
        
        success = rag.delete_document(source_url)
        
        if success:
            return {"message": f"Document {source_url} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_documents(
    query: str,
    k: Optional[int] = 5,
    content_type_filter: Optional[str] = None,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    Search documents in the RAG system.
    Requires authentication.
    """
    try:
        # Verify user authentication
        user = await get_current_user(db, credentials.credentials)
        
        rag = get_rag_system()
        if rag is None:
            raise HTTPException(status_code=500, detail="RAG system not available")
        
        results = rag.vector_store.similarity_search(
            query=query,
            k=k,
            content_type_filter=content_type_filter
        )
        
        search_results = []
        for text, distance, metadata in results:
            search_results.append({
                "text": text,
                "similarity": 1.0 - distance,  # Convert distance to similarity
                "metadata": metadata
            })
        
        return {
            "query": query,
            "results": search_results,
            "total_results": len(search_results)
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag-stats", response_model=SystemStatsResponse)
async def get_system_stats(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    Get RAG system statistics.
    Requires authentication.
    """
    try:
        # Verify user authentication
        user = await get_current_user(db, credentials.credentials)
        
        rag = get_rag_system()
        if rag is None:
            raise HTTPException(status_code=500, detail="RAG system not available")
        
        logger.info("RAG system instance retrieved for status check")
        stats = rag.get_system_stats()
        
        return SystemStatsResponse(
            vector_store=stats.get("vector_store", {}),
            processing_status=stats.get("processing_status", {}),
            total_processed_documents=stats.get("total_processed_documents", 0),
            system_components=stats.get("system_components", {})
        )
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-documents")
async def clear_all_documents(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    Clear all documents from the RAG system.
    Requires authentication.
    """
    try:
        # Verify user authentication
        user = await get_current_user(db, credentials.credentials)
        
        rag = get_rag_system()
        if rag is None:
            raise HTTPException(status_code=500, detail="RAG system not available")
        
        success = rag.clear_all_documents()
        
        if success:
            return {"message": "All documents cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear documents")
            
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
