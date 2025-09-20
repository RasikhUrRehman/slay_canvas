"""
Test Router for Agent and Knowledge Base Management
Provides endpoints for testing knowledge base operations and agent communication
without authentication requirements.
"""

import logging
import time
import tempfile
import os
import io
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from engine.rag.rag_system import RAGSystem
from engine.llm.agent import KnowledgeBaseAgent
from engine.rag.vector_store import VectorStore
from pymilvus import MilvusClient, utility
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/test-agent", tags=["test-agent"])


def _get_knowledge_base(kb_name: str) -> VectorStore:
    """
    Get a knowledge base VectorStore instance by name.
    
    Checks if collection exists in Milvus and returns VectorStore instance.
    """
    # Check if collection exists in Milvus directly
    try:
        # Use the exact collection name (user-provided name)
        collection_name = kb_name.lower().replace(' ', '_')
        temp_vector_store = VectorStore(collection_name="temp_discovery")
        client = temp_vector_store.client
        
        # Check if collection exists
        if client.has_collection(collection_name=collection_name):
            vector_store = VectorStore(collection_name=collection_name)
            return vector_store
                
    except Exception as e:
        logger.warning(f"Error checking knowledge base {kb_name}: {str(e)}")
    
    raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_name}' not found")


# Pydantic models for request/response
class CreateKnowledgeBaseRequest(BaseModel):
    name: str
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


class AgentResponse(BaseModel):
    answer: str
    tools_used: List[str]
    reasoning: str
    confidence: float
    processing_time: float
    sources: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


@router.post("/knowledge-bases", response_model=Dict[str, str])
async def create_knowledge_base(request: CreateKnowledgeBaseRequest):
    """Create a new knowledge base for testing using VectorStore."""
    try:
        # Check if knowledge base already exists
        collection_name = request.name.lower().replace(' ', '_')
        temp_vector_store = VectorStore(collection_name="temp_discovery")
        client = temp_vector_store.client
        
        if client.has_collection(collection_name=collection_name):
            raise HTTPException(status_code=400, detail=f"Knowledge base '{request.name}' already exists")
        
        # Create VectorStore with user-provided name (no test_kb_ prefix)
        vector_store = VectorStore(
            collection_name=collection_name,
            dimension=1536  # OpenAI text-embedding-3-small dimension
        )
        
        logger.info(f"Created knowledge base: {request.name}")
        
        return {
            "message": f"Knowledge base '{request.name}' created successfully",
            "name": request.name,
            "collection_name": collection_name
        }
        
    except Exception as e:
        logger.error(f"Error creating knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create knowledge base: {str(e)}")


@router.get("/knowledge-bases", response_model=List[KnowledgeBaseInfo])
async def list_knowledge_bases():
    """List all available knowledge bases by discovering collections in Milvus."""
    try:
        knowledge_bases = []
        
        # Discover collections using VectorStore
        try:
            # Connect to Milvus through VectorStore to list collections
            temp_vector_store = VectorStore(collection_name="temp_discovery")
            client = temp_vector_store.client
            
            # List all collections in Milvus
            collections = client.list_collections()
            
            for collection_name in collections:
                try:
                    # Skip system collections
                    if collection_name.startswith("temp_") or collection_name in ["temp_discovery"]:
                        continue
                    
                    # Use collection name as knowledge base name
                    kb_name = collection_name.replace("_", " ").title()
                    
                    # Use VectorStore to get collection statistics
                    collection_vector_store = VectorStore(collection_name=collection_name)
                    collection_stats = collection_vector_store.get_collection_stats()
                    documents = collection_vector_store.list_all_documents()
                    
                    # Count unique documents by source_url
                    unique_sources = set()
                    for _, metadata in documents:
                        source_url = metadata.get('source_url', '')
                        if source_url:
                            unique_sources.add(source_url)
                    
                    kb_info = KnowledgeBaseInfo(
                        name=kb_name,
                        description=f"Knowledge base: {kb_name}",
                        document_count=len(unique_sources),
                        chunk_count=collection_stats.get("total_entities", 0),
                        created_at="unknown",
                        stats={
                            "collection_name": collection_name,
                            "total_entities": collection_stats.get("total_entities", 0),
                            "dimension": collection_stats.get("dimension", 0),
                            "unique_sources": len(unique_sources)
                        }
                    )
                    knowledge_bases.append(kb_info)
                    
                except Exception as e:
                    logger.warning(f"Error processing collection {collection_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Error discovering collections: {str(e)}")
        
        return knowledge_bases
        
    except Exception as e:
        logger.error(f"Error listing knowledge bases: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list knowledge bases: {str(e)}")


@router.get("/knowledge-bases/{kb_name}", response_model=KnowledgeBaseInfo)
async def get_knowledge_base(kb_name: str):
    """Get information about a specific knowledge base using VectorStore."""
    try:
        vector_store = _get_knowledge_base(kb_name)
        
        stats = vector_store.get_collection_stats()
        documents = vector_store.list_all_documents()
        
        # Count unique documents by source_url
        unique_sources = set()
        for _, metadata in documents:
            source_url = metadata.get('source_url', '')
            if source_url:
                unique_sources.add(source_url)
        
        return KnowledgeBaseInfo(
            name=kb_name,
            description=f"Test knowledge base: {kb_name}",
            document_count=len(unique_sources),
            chunk_count=stats.get("total_entities", 0),
            created_at="unknown",
            stats=stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting knowledge base {kb_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get knowledge base: {str(e)}")


@router.delete("/knowledge-bases/{kb_name}")
async def delete_knowledge_base(kb_name: str):
    """Delete a knowledge base and all its data using VectorStore."""
    try:
        vector_store = _get_knowledge_base(kb_name)
        
        # Drop the collection completely (this removes it from Milvus)
        collection_name = vector_store.collection_name
        vector_store.client.drop_collection(collection_name=collection_name)
        
        logger.info(f"Deleted knowledge base: {kb_name}")
        return {"message": f"Knowledge base '{kb_name}' deleted successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting knowledge base {kb_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete knowledge base: {str(e)}")


@router.post("/knowledge-bases/{kb_name}/documents/url")
async def add_document_from_url(kb_name: str, request: AddDocumentRequest):
    """Add a document to the knowledge base from a URL using VectorStore."""
    try:
        vector_store = _get_knowledge_base(kb_name)
        
        # Import required services for document processing
        from engine.services.extractor import Extractor
        from engine.rag.document_splitter import DocumentSplitter
        from datetime import datetime
        
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
        metadatas = [chunk.metadata for chunk in chunks]
        
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
async def add_text_document(kb_name: str, request: AddTextRequest):
    """Add a text document to the knowledge base using VectorStore."""
    try:
        vector_store = _get_knowledge_base(kb_name)
        
        # Import required services for document processing
        from engine.rag.document_splitter import DocumentSplitter
        from datetime import datetime
        
        start_time = datetime.now()
        
        # Create metadata with proper structure
        source_id = request.metadata.get('source_url', f"custom_text_{kb_name}_{datetime.now().isoformat()}")
        metadata = {
            "source_url": source_id,
            "title": request.metadata.get('title', 'Custom Text'),
            "content_type": "text/plain",
            "extraction_time": datetime.now().isoformat(),
            "transcription_type": "custom",
            **request.metadata
        }
        
        # Split text into chunks using DocumentSplitter
        splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_custom_text(request.text, metadata)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks could be created from the text")
        
        # Prepare texts and metadata for vector store
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
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
async def add_document_from_file(kb_name: str, file: UploadFile = File(...)):
    """Add a document to the knowledge base from an uploaded file using Extractor."""
    try:
        vector_store = _get_knowledge_base(kb_name)
        
        # Import required services for document processing
        from datetime import datetime
        
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
            # Initialize RAG system with the sanitized collection name
            collection_name = kb_name.lower().replace(' ', '_')
            rag_system = RAGSystem(collection_name=collection_name)
            
            # Process the file using RAG system with extractor
            logger.info(f"Processing uploaded file: {file.filename}")
            
            # Use the new add_document_from_file function that handles file uploads properly
            # This function will use the extractor, handle metadata, and clean up temp files
            status = rag_system.add_document_from_file(temp_file_path, file.filename)
            
            if status.status != "completed":
                error_msg = status.error_message or "Unknown processing error"
                logger.error(f"RAG processing failed for {file.filename}: {error_msg}")
                raise HTTPException(status_code=400, detail=f"Failed to process document: {error_msg}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "message": "Document file added successfully",
                "filename": file.filename,
                "file_type": file_extension,
                "status": "completed",
                "chunks_created": status.chunks_created,
                "processing_time": processing_time,
                "source_url": status.source_url
            }
            
        finally:
            # Note: Temporary file cleanup is now handled automatically by add_document_from_file
            # The function will clean up temp files if they are in temp/tmp directories
            pass
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding document from file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")


@router.post("/knowledge-bases/{kb_name}/documents/audio")
async def add_audio_file(kb_name: str, file: UploadFile = File(...)):
    """Add an audio file to the knowledge base by transcribing it using Extractor."""
    try:
        vector_store = _get_knowledge_base(kb_name)
        
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
async def list_documents(kb_name: str):
    """List all documents in a knowledge base using VectorStore."""
    try:
        vector_store = _get_knowledge_base(kb_name)
        
        # Get all documents from vector store
        documents = vector_store.list_all_documents()
        
        # Group by source URL to create document summaries
        doc_map = {}
        for text, metadata in documents:
            source_url = metadata.get("source_url", "unknown")
            if source_url not in doc_map:
                doc_map[source_url] = {
                    "source_url": source_url,
                    "content_type": metadata.get("content_type", ""),
                    "title": metadata.get("title", ""),
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
async def delete_document(kb_name: str, source_url: str):
    """Delete a specific document from the knowledge base using VectorStore."""
    try:
        vector_store = _get_knowledge_base(kb_name)
        
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


@router.post("/knowledge-bases/{kb_name}/query", response_model=AgentResponse)
async def query_agent(kb_name: str, request: QueryAgentRequest):
    """Query the agent using the specified knowledge base with VectorStore."""
    try:
        # Get the knowledge base
        vector_store = _get_knowledge_base(kb_name)
        
        # Create agent for the knowledge base on demand
        collection_name = vector_store.collection_name
        agent = KnowledgeBaseAgent(rag_collection_name=collection_name)
        
        # Process the query
        start_time = time.time()
        
        # For now, we'll use the agent's existing process_query method
        # Note: The agent internally uses its own RAG system, but this maintains compatibility
        response = agent.process_query(request.message)
        processing_time = time.time() - start_time
        
        return AgentResponse(
            answer=response.answer,
            tools_used=response.tools_used,
            reasoning=response.reasoning,
            confidence=response.confidence,
            processing_time=processing_time,
            sources=response.sources,
            error=response.error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to query agent: {str(e)}")


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