"""
Asset Knowledge Service for linking assets/collections to knowledge bases
and creating chunks in Milvus
"""
import logging
import os
import tempfile
from datetime import datetime
from typing import Dict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.models.asset import Asset
from app.models.collection import Collection
from app.models.knowledge_base import KnowledgeBase
from app.services.knowledge_base_service import knowledge_base_service
from engine.rag.document_splitter import DocumentSplitter
from engine.services.extractor import Extractor

logger = logging.getLogger(__name__)


class AssetKnowledgeService:
    """Service for managing asset-knowledge base relationships and chunking"""
    
    async def link_asset_to_knowledge_base(
        self,
        db: AsyncSession,
        asset_id: int,
        knowledge_base_id: int,
        user_id: int
    ) -> Dict[str, any]:
        """Link an asset to a knowledge base and create chunks in Milvus"""
        
        # Get asset and verify ownership
        asset = await db.get(Asset, asset_id)
        if not asset or asset.user_id != user_id:
            raise ValueError("Asset not found or access denied")
        
        # Get knowledge base and verify ownership
        kb = await db.get(KnowledgeBase, knowledge_base_id)
        if not kb or kb.user_id != user_id:
            raise ValueError("Knowledge base not found or access denied")
        
        # Verify asset and KB are in same workspace
        if asset.workspace_id != kb.workspace_id:
            raise ValueError("Asset and knowledge base must be in the same workspace")
        
        # Link asset to knowledge base
        asset.knowledge_base_id = knowledge_base_id
        
        # Process asset for chunking
        chunk_result = await self._process_asset_for_chunking(asset, kb, db)
        
        await db.commit()
        await db.refresh(asset)
        
        return {
            "message": f"Asset '{asset.title or asset.id}' linked to knowledge base '{kb.name}'",
            "asset_id": asset.id,
            "knowledge_base_id": knowledge_base_id,
            "chunks_created": chunk_result.get("chunks_created", 0),
            "processing_time": chunk_result.get("processing_time", 0)
        }
    
    async def link_collection_to_knowledge_base(
        self,
        db: AsyncSession,
        collection_id: int,
        knowledge_base_id: int,
        user_id: int
    ) -> Dict[str, any]:
        """Link a collection to a knowledge base and create chunks for all assets"""
        
        # Get collection with assets and verify ownership
        query = select(Collection).options(selectinload(Collection.assets)).where(Collection.id == collection_id)
        result = await db.execute(query)
        collection = result.scalar_one_or_none()
        
        if not collection or collection.user_id != user_id:
            raise ValueError("Collection not found or access denied")
        
        # Get knowledge base and verify ownership
        kb = await db.get(KnowledgeBase, knowledge_base_id)
        if not kb or kb.user_id != user_id:
            raise ValueError("Knowledge base not found or access denied")
        
        # Verify collection and KB are in same workspace
        if collection.workspace_id != kb.workspace_id:
            raise ValueError("Collection and knowledge base must be in the same workspace")
        
        # Link collection to knowledge base
        collection.knowledge_base_id = knowledge_base_id
        
        # Process all assets in the collection
        total_chunks = 0
        processed_assets = 0
        errors = []
        
        for asset in collection.assets:
            try:
                chunk_result = await self._process_asset_for_chunking(asset, kb, db)
                total_chunks += chunk_result.get("chunks_created", 0)
                processed_assets += 1
            except Exception as e:
                logger.error(f"Error processing asset {asset.id}: {str(e)}")
                errors.append(f"Asset {asset.id}: {str(e)}")
        
        await db.commit()
        await db.refresh(collection)
        
        return {
            "message": f"Collection '{collection.name}' linked to knowledge base '{kb.name}'",
            "collection_id": collection.id,
            "knowledge_base_id": knowledge_base_id,
            "assets_processed": processed_assets,
            "total_chunks_created": total_chunks,
            "errors": errors
        }
    
    async def unlink_asset_from_knowledge_base(
        self,
        db: AsyncSession,
        asset_id: int,
        user_id: int
    ) -> Dict[str, any]:
        """Unlink an asset from knowledge base and remove chunks from Milvus"""
        
        # Get asset and verify ownership
        asset = await db.get(Asset, asset_id)
        if not asset or asset.user_id != user_id:
            raise ValueError("Asset not found or access denied")
        
        if not asset.knowledge_base_id:
            raise ValueError("Asset is not linked to any knowledge base")
        
        # Get knowledge base
        kb = await db.get(KnowledgeBase, asset.knowledge_base_id)
        if not kb:
            raise ValueError("Associated knowledge base not found")
        
        # Remove chunks from Milvus
        deleted_chunks = await self._remove_asset_chunks(asset, kb)
        
        # Unlink from database
        asset.knowledge_base_id = None
        await db.commit()
        await db.refresh(asset)
        
        return {
            "message": f"Asset '{asset.title or asset.id}' unlinked from knowledge base '{kb.name}'",
            "asset_id": asset.id,
            "chunks_deleted": deleted_chunks
        }
    
    async def unlink_collection_from_knowledge_base(
        self,
        db: AsyncSession,
        collection_id: int,
        user_id: int
    ) -> Dict[str, any]:
        """Unlink collection from knowledge base and remove all asset chunks"""
        
        # Get collection with assets
        query = select(Collection).options(selectinload(Collection.assets)).where(Collection.id == collection_id)
        result = await db.execute(query)
        collection = result.scalar_one_or_none()
        
        if not collection or collection.user_id != user_id:
            raise ValueError("Collection not found or access denied")
        
        if not collection.knowledge_base_id:
            raise ValueError("Collection is not linked to any knowledge base")
        
        # Get knowledge base
        kb = await db.get(KnowledgeBase, collection.knowledge_base_id)
        if not kb:
            raise ValueError("Associated knowledge base not found")
        
        # Remove chunks for all assets
        total_deleted = 0
        for asset in collection.assets:
            try:
                deleted = await self._remove_asset_chunks(asset, kb)
                total_deleted += deleted
            except Exception as e:
                logger.error(f"Error removing chunks for asset {asset.id}: {str(e)}")
        
        # Unlink from database
        collection.knowledge_base_id = None
        await db.commit()
        await db.refresh(collection)
        
        return {
            "message": f"Collection '{collection.name}' unlinked from knowledge base '{kb.name}'",
            "collection_id": collection.id,
            "total_chunks_deleted": total_deleted
        }
    
    async def _process_asset_for_chunking(
        self,
        asset: Asset,
        kb: KnowledgeBase,
        db: AsyncSession
    ) -> Dict[str, any]:
        """Process an individual asset and create chunks in Milvus"""
        start_time = datetime.now()
        
        try:
            # Get vector store for knowledge base
            vector_store = knowledge_base_service.get_vector_store(kb)
            
            # Initialize document splitter
            splitter = DocumentSplitter(
                chunk_size=kb.chunk_size,
                chunk_overlap=kb.chunk_overlap
            )
            
            chunks_created = 0
            
            if asset.type == "text":
                # Process text content directly
                if asset.content:
                    chunks_created = await self._process_text_content(
                        asset, vector_store, splitter
                    )
            
            elif asset.type in ["social", "wiki", "internet"]:
                # Process URL content
                if asset.url:
                    chunks_created = await self._process_url_content(
                        asset, vector_store, splitter
                    )
            
            elif asset.type in ["image", "audio", "document"]:
                # Process file content
                if asset.file_path:
                    chunks_created = await self._process_file_content(
                        asset, vector_store, splitter
                    )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "chunks_created": chunks_created,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing asset {asset.id}: {str(e)}")
            raise
    
    async def _process_text_content(
        self,
        asset: Asset,
        vector_store,
        splitter: DocumentSplitter
    ) -> int:
        """Process text asset content using the same logic as agent.py"""
        
        # Create metadata with proper structure - same as agent.py
        source_id = f"asset_{asset.id}_{datetime.now().isoformat()}"
        metadata = {
            "source_url": source_id,
            "title": asset.title or f"Text Asset {asset.id}",
            "content_type": "text/plain",
            "extraction_time": datetime.now().isoformat(),
            "transcription_type": "custom",
            "user_id": asset.user_id,
            "asset_id": asset.id,
            "asset_type": asset.type,
            "workspace_id": asset.workspace_id,
            **asset.asset_metadata
        }
        
        # Split text into chunks using DocumentSplitter - same as agent.py
        chunks = splitter.split_custom_text(asset.content, metadata)
        
        if not chunks:
            raise ValueError("No chunks could be created from the text")
        
        # Prepare texts and metadata for vector store - same as agent.py
        texts = [chunk.text for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            chunk_metadata = chunk.metadata.copy()
            # Ensure user_id is in metadata
            chunk_metadata['user_id'] = asset.user_id
            metadatas.append(chunk_metadata)
        
        # Add to vector store - same as agent.py
        vector_store.add_documents(texts, metadatas)
        
        return len(chunks)
    
    async def _process_url_content(
        self,
        asset: Asset,
        vector_store,
        splitter: DocumentSplitter
    ) -> int:
        """Process URL-based asset content using the same logic as agent.py"""
        
        # Initialize extractor and splitter - same as agent.py
        extractor = Extractor()
        
        # Extract content from URL - same as agent.py
        logger.info(f"Extracting content from: {asset.url}")
        extracted_content = extractor.process_content(asset.url)
        
        if not extracted_content.get("success", False):
            error_msg = extracted_content.get("error_message", "Unknown extraction error")
            logger.error(f"Extraction failed for {asset.url}: {error_msg}")
            raise ValueError(f"Failed to extract content: {error_msg}")
        
        # Split content into chunks - same as agent.py
        logger.info("Splitting content into chunks...")
        chunks = splitter.split_extracted_content(extracted_content)
        
        if not chunks:
            error_msg = "No content could be extracted for chunking"
            logger.warning(f"No chunks created for {asset.url}: {error_msg}")
            raise ValueError(error_msg)
        
        # Prepare texts and metadata for vector store - same as agent.py
        texts = [chunk.text for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            metadata = chunk.metadata.copy()
            # Add user_id and asset info to metadata
            metadata['user_id'] = asset.user_id
            metadata['asset_id'] = asset.id
            metadata['asset_type'] = asset.type
            metadata['workspace_id'] = asset.workspace_id
            # Merge asset metadata
            metadata.update(asset.asset_metadata or {})
            metadatas.append(metadata)
        
        # Add to vector store - same as agent.py
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        vector_store.add_documents(texts, metadatas)
        
        return len(chunks)
    
    async def _process_file_content(
        self,
        asset: Asset,
        vector_store,
        splitter: DocumentSplitter
    ) -> int:
        """Process file-based asset content using the same logic as agent.py"""
        
        # For audio files, we need transcription
        if asset.type == "audio":
            return await self._process_audio_file(asset, vector_store, splitter)
        
        # For documents and images with text, extract content
        elif asset.type in ["document", "image"]:
            return await self._process_document_file(asset, vector_store, splitter)
        
        return 0
    
    async def _process_audio_file(
        self,
        asset: Asset,
        vector_store,
        splitter: DocumentSplitter
    ) -> int:
        """Process audio file by transcribing it using the same logic as agent.py"""
        
        try:
            # Note: This would need to download the file from MinIO first
            # For now, we'll use a placeholder implementation
            # In the future, we should download the file and process it similar to agent.py
            
            # Get file from MinIO (placeholder - would need actual download)
            logger.info(f"Processing audio file: {asset.file_path}")
            
            # For now, return 0 until we implement MinIO file download
            logger.warning(f"Audio processing requires MinIO file download - not yet implemented for asset {asset.id}")
            return 0
            
        except Exception as e:
            logger.error(f"Error processing audio file for asset {asset.id}: {str(e)}")
            return 0
    
    async def _process_document_file(
        self,
        asset: Asset,
        vector_store,
        splitter: DocumentSplitter
    ) -> int:
        """Process document file by extracting text using the same logic as agent.py"""
        
        try:
            # Note: This would need to download the file from MinIO first
            # For now, we'll use a placeholder implementation
            # In the future, we should download the file and process it similar to agent.py
            
            logger.info(f"Processing document file: {asset.file_path}")
            
            # Validate file type similar to agent.py
            allowed_extensions = {'.pdf', '.docx', '.txt', '.doc'}
            original_filename = asset.asset_metadata.get("original_filename", asset.file_path)
            file_extension = os.path.splitext(original_filename)[1].lower()
            
            if file_extension not in allowed_extensions:
                raise ValueError(f"Unsupported file type: {file_extension}. Allowed types: {', '.join(allowed_extensions)}")
            
            # For now, return 0 until we implement MinIO file download
            logger.warning(f"Document processing requires MinIO file download - not yet implemented for asset {asset.id}")
            return 0
            
        except Exception as e:
            logger.error(f"Error processing document file for asset {asset.id}: {str(e)}")
            return 0
    
    async def _remove_asset_chunks(
        self,
        asset: Asset,
        kb: KnowledgeBase
    ) -> int:
        """Remove chunks for a specific asset from Milvus"""
        
        try:
            # Get vector store
            vector_store = knowledge_base_service.get_vector_store(kb)
            
            # Delete by source_url - need to handle different asset types
            if asset.type == "text":
                # For text assets, we used a timestamped source_url
                # We'll need to delete by asset_id metadata instead
                pass  # Will implement metadata-based deletion
            elif asset.type in ["social", "wiki", "internet"]:
                # For URL assets, source_url is the original URL
                source_url = asset.url
            elif asset.type in ["image", "audio", "document"]:
                # For file assets, source_url is the file_path
                source_url = asset.file_path
            else:
                source_url = f"asset_{asset.id}"
            
            # For now, try to delete by source_url
            # TODO: Implement metadata-based deletion for more robust cleanup
            deleted_count = 0
            if 'source_url' in locals() and source_url:
                deleted_count = vector_store.delete_by_source_url(source_url)
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error removing chunks for asset {asset.id}: {str(e)}")
            return 0


# Create service instance
asset_knowledge_service = AssetKnowledgeService()
