"""
Pydantic schemas for Conversation and Message models
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

# Import MessageRole from the model to ensure consistency
from app.models.message import MessageRole


# Message schemas
class MessageBase(BaseModel):
    content: str
    role: MessageRole


class MessageCreate(MessageBase):
    conversation_id: int
    user_id: Optional[int] = None


class MessageUpdate(BaseModel):
    content: Optional[str] = None


class MessageInDB(MessageBase):
    id: int
    conversation_id: int
    user_id: Optional[int]
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class Message(MessageInDB):
    pass


class MessageRead(MessageInDB):
    pass


# Conversation schemas
class ConversationBase(BaseModel):
    conversation_name: str


class ConversationCreate(ConversationBase):
    project_id: int
    knowledge_base_id: Optional[int] = None


class ConversationUpdate(BaseModel):
    conversation_name: Optional[str] = None


class ConversationInDB(ConversationBase):
    id: int
    project_id: int
    knowledge_base_id: Optional[int] = None
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class Conversation(ConversationInDB):
    pass


class ConversationRead(ConversationInDB):
    messages: Optional[List[MessageRead]] = []


class ConversationWithMessages(ConversationInDB):
    messages: List[MessageRead] = []


# Response schemas
class ConversationResponse(BaseModel):
    message: str
    conversation: ConversationRead


class MessageResponse(BaseModel):
    message: str
    data: MessageRead


# Public schemas for API responses
class MessagePublic(MessageInDB):
    pass


class ConversationPublic(ConversationInDB):
    messages: Optional[List[MessagePublic]] = []