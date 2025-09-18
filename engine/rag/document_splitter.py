"""
Document text splitter module using LangChain for efficient chunking.
Preserves metadata while splitting text into manageable chunks for RAG.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

logger = logging.getLogger(__name__)


@dataclass
class ChunkedDocument:
    """Data class for chunked document with metadata"""
    text: str
    metadata: Dict[str, Any]
    chunk_index: int
    total_chunks: int


class DocumentSplitter:
    """Document text splitter using LangChain with metadata preservation"""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None):
        """
        Initialize the document splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: Custom separators for splitting (defaults to LangChain's)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize different splitters for different content types
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", " ", ""]
        )
        
        self.character_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n"
        )
        
        # Token-based splitter for very large documents
        self.token_splitter = TokenTextSplitter(
            chunk_size=chunk_size // 4,  # Tokens are roughly 4 chars on average
            chunk_overlap=chunk_overlap // 4
        )
    
    def split_extracted_content(self, extracted_content: Dict[str, Any]) -> List[ChunkedDocument]:
        """
        Split extracted content from the extractor service into chunks.
        
        Args:
            extracted_content: Content dict from extractor service with structure:
                {
                    "url": str,
                    "content_type": str,
                    "transcriptions": {
                        "text": str,
                        "audio_transcription": str,
                        "image_transcriptions": [{"url": str, "text": str}]
                    },
                    "metadata": dict,
                    "success": bool,
                    "error_message": str
                }
        
        Returns:
            List of ChunkedDocument objects
        """
        if not extracted_content.get("success", False):
            logger.error(f"Cannot split content due to extraction error: {extracted_content.get('error_message', 'Unknown error')}")
            return []
        
        chunks = []
        base_metadata = {
            "source_url": extracted_content.get("url", ""),
            "content_type": extracted_content.get("content_type", ""),
            "title": extracted_content.get("metadata", {}).get("title", ""),
            "extraction_time": extracted_content.get("metadata", {}).get("extraction_time", ""),
        }
        
        transcriptions = extracted_content.get("transcriptions", {})
        
        # Process main text content
        main_text = transcriptions.get("text", "")
        if main_text.strip():
            text_chunks = self._split_text_content(main_text, "text", base_metadata)
            chunks.extend(text_chunks)
        
        # Process audio transcription
        audio_text = transcriptions.get("audio_transcription", "")
        if audio_text.strip():
            audio_chunks = self._split_text_content(audio_text, "audio", base_metadata)
            chunks.extend(audio_chunks)
        
        # Process image transcriptions
        image_transcriptions = transcriptions.get("image_transcriptions", [])
        for img_trans in image_transcriptions:
            img_text = img_trans.get("text", "")
            if img_text.strip():
                img_metadata = base_metadata.copy()
                img_metadata["image_url"] = img_trans.get("url", "")
                img_chunks = self._split_text_content(img_text, "image", img_metadata)
                chunks.extend(img_chunks)
        
        logger.info(f"Split content into {len(chunks)} chunks from {extracted_content.get('url', 'Unknown')}")
        return chunks
    
    def _split_text_content(self, text: str, transcription_type: str, base_metadata: Dict[str, Any]) -> List[ChunkedDocument]:
        """
        Split text content into chunks using appropriate splitter.
        
        Args:
            text: Text content to split
            transcription_type: Type of transcription (text, audio, image)
            base_metadata: Base metadata to attach to each chunk
            
        Returns:
            List of ChunkedDocument objects
        """
        if not text.strip():
            return []
        
        # Choose splitter based on content characteristics
        splitter = self._choose_splitter(text, transcription_type)
        
        # Split the text
        text_chunks = splitter.split_text(text)
        
        # Create ChunkedDocument objects
        chunked_docs = []
        total_chunks = len(text_chunks)
        
        for i, chunk in enumerate(text_chunks):
            metadata = base_metadata.copy()
            metadata.update({
                "chunk_index": i,
                "total_chunks": total_chunks,
                "chunk_size": len(chunk),
                "transcription_type": transcription_type
            })
            
            chunked_docs.append(ChunkedDocument(
                text=chunk.strip(),
                metadata=metadata,
                chunk_index=i,
                total_chunks=total_chunks
            ))
        
        return chunked_docs
    
    def _choose_splitter(self, text: str, transcription_type: str):
        """
        Choose the appropriate splitter based on text characteristics.
        
        Args:
            text: Text to analyze
            transcription_type: Type of transcription
            
        Returns:
            Appropriate text splitter
        """
        text_length = len(text)
        
        # For very long texts, use token splitter
        if text_length > 50000:
            logger.info(f"Using token splitter for long {transcription_type} content ({text_length} chars)")
            return self.token_splitter
        
        # For audio transcriptions, use character splitter (usually more continuous)
        elif transcription_type == "audio":
            logger.info(f"Using character splitter for audio transcription ({text_length} chars)")
            return self.character_splitter
        
        # For structured text (web pages, documents), use recursive splitter
        else:
            logger.info(f"Using recursive splitter for {transcription_type} content ({text_length} chars)")
            return self.recursive_splitter
    
    def split_custom_text(self, text: str, metadata: Dict[str, Any]) -> List[ChunkedDocument]:
        """
        Split custom text with provided metadata.
        
        Args:
            text: Text to split
            metadata: Metadata to attach to chunks
            
        Returns:
            List of ChunkedDocument objects
        """
        if not text.strip():
            return []
        
        # Use recursive splitter for general text
        text_chunks = self.recursive_splitter.split_text(text)
        
        chunked_docs = []
        total_chunks = len(text_chunks)
        
        for i, chunk in enumerate(text_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": total_chunks,
                "chunk_size": len(chunk),
                "transcription_type": "custom"
            })
            
            chunked_docs.append(ChunkedDocument(
                text=chunk.strip(),
                metadata=chunk_metadata,
                chunk_index=i,
                total_chunks=total_chunks
            ))
        
        return chunked_docs
    
    def merge_small_chunks(self, chunks: List[ChunkedDocument], min_chunk_size: int = 100) -> List[ChunkedDocument]:
        """
        Merge chunks that are smaller than minimum size with adjacent chunks.
        
        Args:
            chunks: List of chunks to process
            min_chunk_size: Minimum chunk size threshold
            
        Returns:
            List of processed chunks
        """
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = None
        
        for chunk in chunks:
            if len(chunk.text) < min_chunk_size and current_chunk is not None:
                # Merge with previous chunk
                merged_text = current_chunk.text + " " + chunk.text
                merged_metadata = current_chunk.metadata.copy()
                merged_metadata["chunk_size"] = len(merged_text)
                
                current_chunk = ChunkedDocument(
                    text=merged_text,
                    metadata=merged_metadata,
                    chunk_index=current_chunk.chunk_index,
                    total_chunks=current_chunk.total_chunks
                )
            else:
                if current_chunk is not None:
                    merged_chunks.append(current_chunk)
                current_chunk = chunk
        
        if current_chunk is not None:
            merged_chunks.append(current_chunk)
        
        # Update total_chunks and chunk_index for merged chunks
        total_merged = len(merged_chunks)
        for i, chunk in enumerate(merged_chunks):
            chunk.chunk_index = i
            chunk.total_chunks = total_merged
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = total_merged
        
        logger.info(f"Merged {len(chunks)} chunks into {len(merged_chunks)} chunks")
        return merged_chunks
    
    def get_chunk_summary(self, chunks: List[ChunkedDocument]) -> Dict[str, Any]:
        """
        Get summary statistics about the chunks.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {"total_chunks": 0, "total_characters": 0, "average_chunk_size": 0}
        
        total_chars = sum(len(chunk.text) for chunk in chunks)
        avg_size = total_chars / len(chunks)
        
        transcription_types = {}
        for chunk in chunks:
            trans_type = chunk.metadata.get("transcription_type", "unknown")
            transcription_types[trans_type] = transcription_types.get(trans_type, 0) + 1
        
        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "average_chunk_size": avg_size,
            "transcription_types": transcription_types,
            "size_distribution": {
                "min": min(len(chunk.text) for chunk in chunks),
                "max": max(len(chunk.text) for chunk in chunks),
                "median": sorted([len(chunk.text) for chunk in chunks])[len(chunks) // 2]
            }
        }
