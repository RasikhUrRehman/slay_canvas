"""
Asset Knowledge Service for linking assets/collections to knowledge bases
and creating chunks in Milvus
"""
import logging
from datetime import datetime
from typing import Dict, Optional

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
        user_id: int,
        asset_handle: Optional[str] = None,
        kb_handle: Optional[str] = None
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
        
        # Save handle information if provided
        if asset_handle is not None:
            asset.kb_connection_asset_handle = asset_handle
        if kb_handle is not None:
            asset.kb_connection_kb_handle = kb_handle
        
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
        user_id: int,
        collection_handle: Optional[str] = None,
        kb_handle: Optional[str] = None
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
        
        # Save handle information if provided
        if collection_handle is not None:
            collection.kb_connection_asset_handle = collection_handle
        if kb_handle is not None:
            collection.kb_connection_kb_handle = kb_handle
        
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
            
            elif asset.type in ["image", "audio", "document", "video"]:
                # Process file content
                if asset.file_path:
                    chunks_created = await self._process_file_content(
                        asset, kb, vector_store, splitter
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
        kb: KnowledgeBase,
        vector_store,
        splitter: DocumentSplitter
    ) -> int:
        """Process file-based asset content using the same logic as agent.py"""
        
        # For audio files, we need transcription
        if asset.type == "audio":
            return await self._process_audio_file(asset, kb, vector_store, splitter)
        
        # For video files, we need transcription (similar to audio)
        elif asset.type == "video":
            return await self._process_video_file(asset, kb, vector_store, splitter)
        
        # For documents and images with text, extract content
        elif asset.type in ["document", "image"]:
            return await self._process_document_file(asset, kb, vector_store, splitter)
        
        return 0
    
    async def _process_video_file(
        self,
        asset: Asset,
        kb: KnowledgeBase,
        vector_store,
        splitter: DocumentSplitter
    ) -> int:
        """Process video file by transcribing it using the same logic as agent.py"""
        
        try:
            logger.info(f"Processing video file: {asset.file_path}")
            
            # Initialize extractor - same as agent.py
            extractor = Extractor()
            
            # For Cloudinary URLs or direct URLs, use extractor.process_content
            logger.info(f"Extracting content from: {asset.file_path}")
            extracted_content = extractor.process_content(asset.file_path)
            
            if not extracted_content.get("success", False):
                error_msg = extracted_content.get("error_message", "Unknown video processing error")
                logger.error(f"Video processing failed for {asset.file_path}: {error_msg}")
                raise ValueError(f"Failed to process video: {error_msg}")
            
            # Override the source_url to use the asset title instead of temp path
            extracted_content["url"] = asset.title or f"Video Asset {asset.id}"
            if "metadata" not in extracted_content:
                extracted_content["metadata"] = {}
            extracted_content["metadata"]["title"] = asset.title or f"Video Asset {asset.id}"
            extracted_content["metadata"]["original_filename"] = asset.title or f"Video Asset {asset.id}"
            
            # Split content into chunks using document splitter
            logger.info(f"Splitting video content into chunks...")
            chunks = splitter.split_extracted_content(extracted_content)
            
            if not chunks:
                error_msg = "No content could be extracted for chunking from video"
                logger.warning(f"No chunks created for {asset.file_path}: {error_msg}")
                raise ValueError(error_msg)
            
            # Prepare texts and metadata for batch insertion
            texts = [chunk.text for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                metadata = chunk.metadata.copy()
                metadata.update({
                    "source_url": asset.file_path,
                    "content_type": "video",
                    "original_filename": asset.title or f"Video Asset {asset.id}",
                    "processing_method": "extractor_video_processing",
                    "user_id": asset.user_id,
                    "asset_id": asset.id,
                    "asset_type": asset.type,
                    "workspace_id": asset.workspace_id,
                    **asset.asset_metadata
                })
                metadatas.append(metadata)
            
            # Add chunks to vector store
            logger.info(f"Adding {len(chunks)} chunks to vector store...")
            document_ids = vector_store.add_documents(texts, metadatas)
            
            logger.info(f"Successfully processed video file: {asset.file_path} with {len(chunks)} chunks")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error processing video file {asset.file_path}: {str(e)}")
            raise ValueError(f"Failed to process video file: {str(e)}")
    
    async def _process_audio_file(
        self,
        asset: Asset,
        kb: KnowledgeBase,
        vector_store,
        splitter: DocumentSplitter
    ) -> int:
        """Process audio file by transcribing it using the same logic as agent.py"""
        
        try:
            logger.info(f"Processing audio file: {asset.file_path}")
            
            # Initialize extractor - same as agent.py
            extractor = Extractor()
            
            # For Cloudinary URLs or direct URLs, use extractor.process_content
            # For local files, also use extractor.process_content
            logger.info(f"Extracting content from: {asset.file_path}")
            extracted_content = extractor.process_content(asset.file_path)
            
            if not extracted_content.get("success", False):
                error_msg = extracted_content.get("error_message", "Unknown extraction error")
                logger.error(f"Audio extraction failed for {asset.file_path}: {error_msg}")
                raise ValueError(f"Failed to extract audio content: {error_msg}")
            
            # Get the audio transcription - same as agent.py
            transcriptions = extracted_content.get("transcriptions", {})
            audio_transcription = transcriptions.get("audio_transcription", "")
            
            if not audio_transcription:
                error_msg = "No audio transcription could be extracted"
                logger.warning(f"No transcription for {asset.file_path}: {error_msg}")
                raise ValueError(error_msg)
            
            # Create metadata with proper structure - same as agent.py
            metadata = {
                "source_url": asset.file_path,
                "title": asset.title or extracted_content.get("title", f"Audio Asset {asset.id}"),
                "content_type": extracted_content.get("content_type", "audio"),
                "extraction_time": datetime.now().isoformat(),
                "transcription_type": "audio",
                "user_id": asset.user_id,
                "asset_id": asset.id,
                "asset_type": asset.type,
                "workspace_id": asset.workspace_id,
                **extracted_content.get("metadata", {}),
                **asset.asset_metadata
            }
            
            # Split transcription into chunks - same as agent.py
            logger.info("Splitting audio transcription into chunks...")
            chunks = splitter.split_custom_text(audio_transcription, metadata)
            
            if not chunks:
                error_msg = "No chunks could be created from audio transcription"
                logger.warning(f"No chunks created for {asset.file_path}: {error_msg}")
                raise ValueError(error_msg)
            
            # Prepare texts and metadata for vector store - same as agent.py
            texts = [chunk.text for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                chunk_metadata = chunk.metadata.copy()
                # Ensure user_id is in metadata
                chunk_metadata['user_id'] = asset.user_id
                metadatas.append(chunk_metadata)
            
            # Add to vector store - same as agent.py
            logger.info(f"Adding {len(chunks)} chunks to vector store...")
            vector_store.add_documents(texts, metadatas)
            
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error processing audio file for asset {asset.id}: {str(e)}")
            raise
    
    async def _process_document_file(
        self,
        asset: Asset,
        kb: KnowledgeBase,
        vector_store,
        splitter: DocumentSplitter
    ) -> int:
        """Process document or image file by downloading from Cloudinary and extracting content using the extractor"""
        
        try:
            import os
            import tempfile

            import requests

            from engine.services.extractor import Extractor
            
            logger.info(f"Processing file: {asset.file_path}")
            
            # Download the file from Cloudinary URL to a temporary file
            logger.info(f"Downloading file from: {asset.file_path}")
            response = requests.get(asset.file_path, timeout=30)
            response.raise_for_status()
            
            # Get file extension from original filename or URL
            original_filename = asset.asset_metadata.get("original_filename", "")
            if original_filename:
                file_extension = os.path.splitext(original_filename)[1].lower()
            else:
                # Try to get extension from URL
                file_extension = os.path.splitext(asset.file_path)[1].lower()
                if not file_extension:
                    file_extension = '.pdf'  # Default to PDF
            
            # Validate file type - now includes both documents and images
            document_extensions = {'.pdf', '.docx', '.txt', '.doc'}
            image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'}
            allowed_extensions = document_extensions | image_extensions
            
            if file_extension not in allowed_extensions:
                raise ValueError(f"Unsupported file type: {file_extension}. Allowed types: {', '.join(sorted(allowed_extensions))}")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            try:
                # Initialize extractor
                extractor = Extractor()
                
                # Process the downloaded file using extractor
                logger.info("Processing file with extractor...")
                extraction_result = extractor.process_content(temp_file_path)
                
                if not extraction_result or not extraction_result.get("success") or not extraction_result.get("transcriptions", {}).get("text"):
                    error_msg = "No content could be extracted from file"
                    logger.warning(f"No content extracted for {asset.file_path}: {error_msg}")
                    raise ValueError(error_msg)
                
                # Split the extracted content into chunks using DocumentSplitter
                chunked_documents = splitter.split_extracted_content(extraction_result)
                
                if not chunked_documents:
                    error_msg = "No chunks created from extracted content"
                    logger.warning(f"No chunks created for {asset.file_path}: {error_msg}")
                    raise ValueError(error_msg)
                
                # Prepare texts and metadata for vector store
                texts = []
                metadatas = []
                
                # Determine content type based on file extension
                content_type = "image" if file_extension in image_extensions else "document"
                
                for chunk_doc in chunked_documents:
                    # Enhance the chunk metadata with asset-specific information
                    enhanced_metadata = {
                        "source_url": asset.file_path,
                        "title": asset.title or extraction_result.get("title") or f"{content_type.title()} Asset {asset.id}",
                        "content_type": content_type,
                        "extraction_time": datetime.now().isoformat(),
                        "user_id": asset.user_id,
                        "asset_id": asset.id,
                        "asset_type": asset.type,
                        "workspace_id": asset.workspace_id,
                        # Add asset metadata
                        **asset.asset_metadata,
                        # Add chunk metadata from DocumentSplitter
                        **chunk_doc.metadata
                    }
                    
                    texts.append(chunk_doc.text)
                    metadatas.append(enhanced_metadata)
                
                # Add to vector store
                logger.info(f"Adding {len(texts)} chunks to vector store...")
                vector_store.add_documents(texts, metadatas)
                
                return len(texts)
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass
            
        except Exception as e:
            logger.error(f"Error processing file for asset {asset.id}: {str(e)}")
            raise
    
    async def _remove_asset_chunks(
        self,
        asset: Asset,
        kb: KnowledgeBase
    ) -> int:
        """Remove chunks for a specific asset from Milvus - 100% replication of agent.py logic"""
        
        try:
            # Get vector store - same as agent.py: kb, vector_store = await _get_knowledge_base_from_db(kb_name, current_user_id, db)
            vector_store = knowledge_base_service.get_vector_store(kb)
            
            # Determine source_url based on asset type - exactly like agent.py expects
            source_url = None
            if asset.type == "text":
                # For text assets, we used a timestamped source_url like: f"asset_{asset.id}_{datetime.now().isoformat()}"
                # Since we can't recreate the exact timestamp, we'll try the base pattern
                source_url = f"asset_{asset.id}"
            elif asset.type in ["social", "wiki", "internet"]:
                # For URL assets, source_url is the original URL
                source_url = asset.url
            elif asset.type in ["image", "audio", "document"]:
                # For file assets, source_url is the file_path
                source_url = asset.file_path
            else:
                source_url = f"asset_{asset.id}"
            
            # EXACT replication of agent.py deletion logic
            if source_url:
                # Delete document by source URL - EXACTLY like agent.py
                deleted_count = vector_store.delete_by_source_url(source_url)
                
                if deleted_count > 0:
                    logger.info(f"Document '{source_url}' deleted successfully, chunks_deleted: {deleted_count}")
                    return deleted_count
                else:
                    logger.warning(f"Document '{source_url}' not found")
                    return 0
            else:
                logger.warning(f"No source_url found for asset {asset.id}")
                return 0
            
        except Exception as e:
            # EXACTLY like agent.py error handling
            logger.error(f"Error deleting document: {str(e)}")
            return 0


# Create service instance
asset_knowledge_service = AssetKnowledgeService()
