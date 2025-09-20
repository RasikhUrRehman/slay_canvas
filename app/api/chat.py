"""
Chat router for streaming conversations with RAG system.
Integrates with the existing Slay Canvas authentication and infrastructure.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.db.session import get_db
from app.services.auth_service import get_current_user
from app.models.user import User
from app.services.conversation_service import conversation_service, message_service
from app.schemas.conversation import (
    ConversationCreate, ConversationUpdate, ConversationPublic,
    MessageCreate, MessageUpdate, MessagePublic
)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from engine.rag_system import RAGSystem
    from engine.services.openrouter import OpenRouterClient
    RAG_AVAILABLE = True
except ImportError:
    # Fallback if engine is not available
    RAGSystem = None
    OpenRouterClient = None
    RAG_AVAILABLE = False

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/chat", tags=["chat"])
security = HTTPBearer()

# Global instances (in production, use dependency injection)
rag_system = None
llm_client = None


# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    model: Optional[str] = Field(default="openai/gpt-3.5-turbo", description="Model to use")
    max_tokens: Optional[int] = Field(default=1000, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    stream: Optional[bool] = Field(default=True, description="Whether to stream response")
    use_rag: Optional[bool] = Field(default=True, description="Whether to use RAG context")
    rag_k: Optional[int] = Field(default=5, description="Number of RAG chunks to retrieve")
    content_type_filter: Optional[str] = Field(default=None, description="Filter by content type")


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None


class StreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


def get_rag_system():
    """Get or initialize RAG system."""
    global rag_system
    
    if not RAG_AVAILABLE:
        logger.warning("RAG system not available - engine module not found")
        return None
        
    if rag_system is None:
        try:
            rag_system = RAGSystem(
                collection_name="chat_documents",
                chunk_size=1000,
                chunk_overlap=200,
                top_k=5
            )
            logger.info("RAG system initialized for chat")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return None
    return rag_system


def get_llm_client(model: str):
    """Get or initialize LLM client."""
    global llm_client
    
    if not RAG_AVAILABLE:
        logger.warning("OpenRouter client not available - engine module not found")
        return None
        
    if llm_client is None or llm_client.model != model:
        try:
            llm_client = OpenRouterClient(
                model=model,
                system_prompt="You are a helpful AI assistant. Use the provided context to answer questions accurately and concisely."
            )
            logger.info(f"LLM client initialized with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            return None
    return llm_client


def enhance_messages_with_rag(messages: List[ChatMessage], rag_k: int = 5, content_type_filter: Optional[str] = None) -> List[Dict[str, str]]:
    """Enhance messages with RAG context."""
    try:
        rag = get_rag_system()
        if rag is None:
            # Fallback to original messages if RAG is not available
            return [{"role": msg.role, "content": msg.content} for msg in messages]
        
        # Get the last user message for context retrieval
        user_messages = [msg for msg in messages if msg.role == "user"]
        if not user_messages:
            # No user messages, return original
            return [{"role": msg.role, "content": msg.content} for msg in messages]
        
        last_user_message = user_messages[-1].content
        
        # Search for relevant context
        search_results = rag.vector_store.similarity_search(
            query=last_user_message,
            k=rag_k,
            content_type_filter=content_type_filter
        )
        
        if search_results:
            # Build context
            context_parts = []
            for text, distance, metadata in search_results:
                source = metadata.get("source_url", "Unknown source")
                context_parts.append(f"Source: {source}\n{text}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Create enhanced messages
            enhanced_messages = []
            
            # Add system message with context
            system_content = f"""You are a helpful AI assistant. Use the following context to answer the user's questions accurately and concisely. If the context doesn't contain relevant information, say so and provide a general response.

Context:
{context}

Please answer based on the context provided above."""
            
            enhanced_messages.append({"role": "system", "content": system_content})
            
            # Add conversation history (excluding any existing system messages)
            for msg in messages:
                if msg.role != "system":
                    enhanced_messages.append({"role": msg.role, "content": msg.content})
            
            return enhanced_messages
        else:
            # No context found, return original messages
            return [{"role": msg.role, "content": msg.content} for msg in messages]
            
    except Exception as e:
        logger.error(f"Error enhancing messages with RAG: {e}")
        # Fallback to original messages
        return [{"role": msg.role, "content": msg.content} for msg in messages]


@router.post("/completions")
async def chat_completions(
    request: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a chat completion with optional RAG enhancement.
    Supports both streaming and non-streaming responses.
    Requires authentication.
    """
    try:
        # Verify user authentication
        user = await get_current_user(db, credentials.credentials)
        
        # Prepare messages
        if request.use_rag:
            messages = enhance_messages_with_rag(
                request.messages, 
                request.rag_k, 
                request.content_type_filter
            )
        else:
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Get LLM client
        client = get_llm_client(request.model)
        if client is None:
            raise HTTPException(status_code=500, detail="LLM client not available")
        
        if request.stream:
            # Return streaming response
            return StreamingResponse(
                stream_chat_response(client, messages, request),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # Return non-streaming response
            response = client.chat_completion(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            if "error" in response:
                raise HTTPException(status_code=500, detail=response["error"])
            
            return ChatResponse(
                id=f"chatcmpl-{datetime.now().timestamp()}",
                created=int(datetime.now().timestamp()),
                model=request.model,
                choices=response["choices"],
                usage=response.get("usage")
            )
            
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Conversation Management Endpoints

@router.post("/conversations", response_model=ConversationPublic)
async def create_conversation(
    conversation_data: ConversationCreate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Create a new conversation."""
    try:
        user = await get_current_user(db, credentials.credentials)
        conversation = await conversation_service.create_conversation(
            db, conversation_data, user.id
        )
        return ConversationPublic.from_orm(conversation)
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations", response_model=List[ConversationPublic])
async def get_conversations(
    project_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Get conversations for the current user."""
    try:
        user = await get_current_user(db, credentials.credentials)
        
        if project_id:
            conversations = await conversation_service.get_conversations_by_project(
                db, project_id, user.id, skip, limit
            )
        else:
            conversations = await conversation_service.get_conversations_by_user(
                db, user.id, skip, limit
            )
        
        return [ConversationPublic.from_orm(conv) for conv in conversations]
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}", response_model=ConversationPublic)
async def get_conversation(
    conversation_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific conversation with messages."""
    try:
        user = await get_current_user(db, credentials.credentials)
        conversation = await conversation_service.get_conversation_by_id(db, conversation_id)
        
        if not conversation or conversation.user_id != user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return ConversationPublic.from_orm(conversation)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/conversations/{conversation_id}", response_model=ConversationPublic)
async def update_conversation(
    conversation_id: int,
    conversation_data: ConversationUpdate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Update a conversation."""
    try:
        user = await get_current_user(db, credentials.credentials)
        conversation = await conversation_service.update_conversation(
            db, conversation_id, conversation_data, user.id
        )
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return ConversationPublic.from_orm(conversation)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Delete a conversation."""
    try:
        user = await get_current_user(db, credentials.credentials)
        success = await conversation_service.delete_conversation(db, conversation_id, user.id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {"message": "Conversation deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Message Management Endpoints

@router.post("/conversations/{conversation_id}/messages", response_model=MessagePublic)
async def create_message(
    conversation_id: int,
    message_data: MessageCreate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Create a new message in a conversation."""
    try:
        user = await get_current_user(db, credentials.credentials)
        
        # Verify conversation exists and belongs to user
        conversation = await conversation_service.get_conversation_by_id(db, conversation_id)
        if not conversation or conversation.user_id != user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Set conversation_id and user_id
        message_data.conversation_id = conversation_id
        message_data.user_id = user.id
        
        message = await message_service.create_message(db, message_data)
        return MessagePublic.from_orm(message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessagePublic])
async def get_messages(
    conversation_id: int,
    skip: int = 0,
    limit: int = 100,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Get messages for a conversation."""
    try:
        user = await get_current_user(db, credentials.credentials)
        
        # Verify conversation exists and belongs to user
        conversation = await conversation_service.get_conversation_by_id(db, conversation_id)
        if not conversation or conversation.user_id != user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        messages = await message_service.get_messages_by_conversation(
            db, conversation_id, skip, limit
        )
        return [MessagePublic.from_orm(msg) for msg in messages]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/messages/{message_id}", response_model=MessagePublic)
async def update_message(
    message_id: int,
    message_data: MessageUpdate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Update a message."""
    try:
        user = await get_current_user(db, credentials.credentials)
        message = await message_service.update_message(db, message_id, message_data, user.id)
        
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        return MessagePublic.from_orm(message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/messages/{message_id}")
async def delete_message(
    message_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Delete a message."""
    try:
        user = await get_current_user(db, credentials.credentials)
        success = await message_service.delete_message(db, message_id, user.id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Message not found")
        
        return {"message": "Message deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_chat_response(client, messages: List[Dict[str, str]], request: ChatRequest):
    """Generate streaming chat response."""
    try:
        # Use the raw streaming method
        for chunk in client.chat_stream_raw(
            messages=messages,
            max_tokens=request.max_tokens
        ):
            if chunk:
                # Format as SSE (Server-Sent Events)
                chunk_data = {
                    "id": f"chatcmpl-{datetime.now().timestamp()}",
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
        
        # Send final chunk
        final_chunk = {
            "id": f"chatcmpl-{datetime.now().timestamp()}",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


@router.get("/models")
async def list_models(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """List available models."""
    try:
        # Verify user authentication
        user = await get_current_user(db, credentials.credentials)
        
        # Default models (you can expand this)
        models = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-3-haiku",
            "anthropic/claude-3-sonnet",
            "mistralai/mistral-7b-instruct",
            "meta-llama/llama-3-8b-instruct",
        ]
        
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "openrouter"
                }
                for model in models
            ]
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag-query")
async def rag_query(
    query: str,
    k: Optional[int] = 5,
    content_type_filter: Optional[str] = None,
    generate_answer: Optional[bool] = True,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    Query the RAG system directly.
    """
    try:
        # Verify user authentication
        user = await get_current_user(db, credentials.credentials)
        
        rag = get_rag_system()
        if rag is None:
            raise HTTPException(status_code=500, detail="RAG system not available")
        
        response = rag.query(
            question=query,
            k=k,
            content_type_filter=content_type_filter,
            generate_answer=generate_answer
        )
        
        return {
            "query": response.query,
            "answer": response.answer,
            "confidence": response.confidence,
            "processing_time": response.processing_time,
            "sources": response.sources,
            "error": response.error
        }
        
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag-stats")
async def rag_stats(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Get RAG system statistics."""
    try:
        # Verify user authentication
        user = await get_current_user(db, credentials.credentials)
        
        rag = get_rag_system()
        if rag is None:
            raise HTTPException(status_code=500, detail="RAG system not available")
        
        stats = rag.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
