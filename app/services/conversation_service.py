"""
Service layer for conversation and message operations
"""
from typing import List, Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.conversation import Conversation
from app.models.message import Message, MessageRole
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    MessageCreate,
    MessageUpdate,
)


class ConversationService:
    async def get_conversation_by_id(self, db: AsyncSession, conversation_id: int) -> Optional[Conversation]:
        """Get conversation by ID with messages."""
        result = await db.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .options(selectinload(Conversation.messages))
        )
        return result.scalar_one_or_none()
    
    async def get_conversations_by_project(self, db: AsyncSession, project_id: int, user_id: int, skip: int = 0, limit: int = 100) -> List[Conversation]:
        """Get conversations for a project by user."""
        result = await db.execute(
            select(Conversation)
            .where(Conversation.project_id == project_id, Conversation.user_id == user_id)
            .order_by(desc(Conversation.updated_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_conversations_by_user(self, db: AsyncSession, user_id: int, skip: int = 0, limit: int = 100) -> List[Conversation]:
        """Get all conversations for a user."""
        result = await db.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(desc(Conversation.updated_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_conversations_by_knowledge_base(self, db: AsyncSession, knowledge_base_id: int, user_id: int, skip: int = 0, limit: int = 100) -> List[Conversation]:
        """Get conversations for a knowledge base by user."""
        result = await db.execute(
            select(Conversation)
            .where(Conversation.knowledge_base_id == knowledge_base_id, Conversation.user_id == user_id)
            .order_by(desc(Conversation.updated_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def create_conversation(self, db: AsyncSession, conversation_data: ConversationCreate, user_id: int) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            conversation_name=conversation_data.conversation_name,
            project_id=conversation_data.project_id,
            knowledge_base_id=conversation_data.knowledge_base_id,
            user_id=user_id
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
        return conversation
    
    async def update_conversation(self, db: AsyncSession, conversation_id: int, conversation_data: ConversationUpdate, user_id: int) -> Optional[Conversation]:
        """Update a conversation."""
        result = await db.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        )
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            return None
        
        if conversation_data.conversation_name is not None:
            conversation.conversation_name = conversation_data.conversation_name
        
        await db.commit()
        await db.refresh(conversation)
        return conversation
    
    async def delete_conversation(self, db: AsyncSession, conversation_id: int, user_id: int) -> bool:
        """Delete a conversation."""
        result = await db.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        )
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            return False
        
        await db.delete(conversation)
        await db.commit()
        return True


class MessageService:
    async def get_message_by_id(self, db: AsyncSession, message_id: int) -> Optional[Message]:
        """Get message by ID."""
        result = await db.execute(select(Message).where(Message.id == message_id))
        return result.scalar_one_or_none()
    
    async def get_messages_by_conversation(self, db: AsyncSession, conversation_id: int, skip: int = 0, limit: int = 100) -> List[Message]:
        """Get messages for a conversation."""
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def create_message(self, db: AsyncSession, message_data: MessageCreate) -> Message:
        """Create a new message."""
        message = Message(
            content=message_data.content,
            role=message_data.role,
            conversation_id=message_data.conversation_id,
            user_id=message_data.user_id
        )
        db.add(message)
        await db.commit()
        await db.refresh(message)
        return message
    
    async def update_message(self, db: AsyncSession, message_id: int, message_data: MessageUpdate, user_id: int = None) -> Optional[Message]:
        """Update a message."""
        query = select(Message).where(Message.id == message_id)
        if user_id:
            query = query.where(Message.user_id == user_id)
        
        result = await db.execute(query)
        message = result.scalar_one_or_none()
        
        if not message:
            return None
        
        if message_data.content is not None:
            message.content = message_data.content
        
        await db.commit()
        await db.refresh(message)
        return message
    
    async def delete_message(self, db: AsyncSession, message_id: int, user_id: int = None) -> bool:
        """Delete a message."""
        query = select(Message).where(Message.id == message_id)
        if user_id:
            query = query.where(Message.user_id == user_id)
        
        result = await db.execute(query)
        message = result.scalar_one_or_none()
        
        if not message:
            return False
        
        await db.delete(message)
        await db.commit()
        return True


# Service instances
conversation_service = ConversationService()
message_service = MessageService()