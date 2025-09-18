"""
Comprehensive RAG (Retrieval-Augmented Generation) System
Integrates extraction, chunking, embedding, and retrieval for multi-source content.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json

# Import custom modules
from engine.services.extractor import Extractor
from engine.document_splitter import DocumentSplitter, ChunkedDocument
from engine.vector_store_new import VectorStore
from engine.services.openrouter import OpenRouterClient

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG query"""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    confidence: float
    processing_time: float
    error: Optional[str] = None


@dataclass
class DocumentStatus:
    """Status of document processing"""
    source_url: str
    status: str  # "processing", "completed", "error"
    chunks_created: int
    processing_time: float
    error_message: Optional[str] = None
    extraction_metadata: Optional[Dict[str, Any]] = None


class RAGSystem:
    """Comprehensive RAG system for multi-source content processing and retrieval"""
    
    def __init__(self, 
                 collection_name: str = "rag_documents",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 top_k: int = 5):
        """
        Initialize the RAG system.
        
        Args:
            collection_name: Name for the Milvus collection
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Default number of results to retrieve
        """
        self.collection_name = collection_name
        self.top_k = top_k
        
        # Initialize components
        logger.info("Initializing RAG system components...")
        
        try:
            # Initialize extractor
            self.extractor = Extractor(headless=True, timeout=30)
            logger.info("✓ Extractor initialized")
            
            # Initialize document splitter
            self.splitter = DocumentSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            logger.info("✓ Document splitter initialized")
            
            # Initialize vector store
            self.vector_store = VectorStore(
                collection_name=collection_name,
                dimension=768  # NLP Cloud embedding dimension
            )
            logger.info("✓ Vector store initialized")
            
            # Initialize OpenRouter client for generation
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            openrouter_model = os.getenv("OPENROUTER_MODEL")

            if openrouter_api_key:
                self.llm_client = OpenRouterClient(model=openrouter_model, api_key=openrouter_api_key)
                logger.info("✓ OpenRouter client initialized")
            else:
                logger.warning("OpenRouter API key not found. Generation features will be limited.")
                self.llm_client = None
            
            # Track processing status
            self.processing_status = {}
            
            logger.info("🚀 RAG system initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def add_document_from_url(self, url: str) -> DocumentStatus:
        """
        Add a document from URL to the RAG system.
        
        Args:
            url: URL to extract and add
            
        Returns:
            DocumentStatus object with processing results
        """
        start_time = datetime.now()
        logger.info(f"Processing document from URL: {url}")
        
        try:
            # Update status
            self.processing_status[url] = DocumentStatus(
                source_url=url,
                status="processing",
                chunks_created=0,
                processing_time=0.0
            )
            
            # Extract content
            logger.info(f"Extracting content from: {url}")
            extracted_content = self.extractor.process_content(url)
            
            if not extracted_content.get("success", False):
                error_msg = extracted_content.get("error_message", "Unknown extraction error")
                logger.error(f"Extraction failed for {url}: {error_msg}")
                
                self.processing_status[url].status = "error"
                self.processing_status[url].error_message = error_msg
                self.processing_status[url].processing_time = (datetime.now() - start_time).total_seconds()
                
                return self.processing_status[url]
            
            # Split content into chunks
            logger.info(f"Splitting content into chunks...")
            chunks = self.splitter.split_extracted_content(extracted_content)
            
            if not chunks:
                error_msg = "No content could be extracted for chunking"
                logger.warning(f"No chunks created for {url}: {error_msg}")
                
                self.processing_status[url].status = "error"
                self.processing_status[url].error_message = error_msg
                self.processing_status[url].processing_time = (datetime.now() - start_time).total_seconds()
                
                return self.processing_status[url]
            
            # Prepare texts and metadata for vector store
            texts = [chunk.text for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Add to vector store
            logger.info(f"Adding {len(chunks)} chunks to vector store...")
            document_ids = self.vector_store.add_documents(texts, metadatas)
            
            # Update status
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.processing_status[url] = DocumentStatus(
                source_url=url,
                status="completed",
                chunks_created=len(chunks),
                processing_time=processing_time,
                extraction_metadata=extracted_content.get("metadata", {})
            )
            
            logger.info(f"✓ Successfully processed {url}: {len(chunks)} chunks in {processing_time:.2f}s")
            return self.processing_status[url]
            
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            logger.error(f"Failed to process {url}: {error_msg}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.processing_status[url] = DocumentStatus(
                source_url=url,
                status="error",
                chunks_created=0,
                processing_time=processing_time,
                error_message=error_msg
            )
            
            return self.processing_status[url]
    
    def add_custom_text(self, text: str, metadata: Dict[str, Any]) -> DocumentStatus:
        """
        Add custom text content to the RAG system.
        
        Args:
            text: Text content to add
            metadata: Metadata for the text
            
        Returns:
            DocumentStatus object with processing results
        """
        start_time = datetime.now()
        source_id = metadata.get("source_url", f"custom_text_{datetime.now().isoformat()}")
        
        logger.info(f"Processing custom text: {source_id}")
        
        try:
            # Split text into chunks
            chunks = self.splitter.split_custom_text(text, metadata)
            
            if not chunks:
                error_msg = "No chunks could be created from the text"
                logger.warning(f"No chunks created for custom text: {error_msg}")
                
                return DocumentStatus(
                    source_url=source_id,
                    status="error",
                    chunks_created=0,
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    error_message=error_msg
                )
            
            # Prepare texts and metadata for vector store
            texts = [chunk.text for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Add to vector store
            logger.info(f"Adding {len(chunks)} chunks to vector store...")
            document_ids = self.vector_store.add_documents(texts, metadatas)
            
            # Create status
            processing_time = (datetime.now() - start_time).total_seconds()
            status = DocumentStatus(
                source_url=source_id,
                status="completed",
                chunks_created=len(chunks),
                processing_time=processing_time
            )
            
            self.processing_status[source_id] = status
            
            logger.info(f"✓ Successfully processed custom text: {len(chunks)} chunks in {processing_time:.2f}s")
            return status
            
        except Exception as e:
            error_msg = f"Error processing custom text: {str(e)}"
            logger.error(f"Failed to process custom text: {error_msg}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            status = DocumentStatus(
                source_url=source_id,
                status="error",
                chunks_created=0,
                processing_time=processing_time,
                error_message=error_msg
            )
            
            self.processing_status[source_id] = status
            return status
    
    def query(self, 
              question: str, 
              k: Optional[int] = None,
              content_type_filter: Optional[str] = None,
              generate_answer: bool = True) -> RAGResponse:
        """
        Query the RAG system with a question.
        
        Args:
            question: Question to ask
            k: Number of chunks to retrieve (defaults to self.top_k)
            content_type_filter: Optional filter by content type
            generate_answer: Whether to generate an answer using LLM
            
        Returns:
            RAGResponse with answer and sources
        """
        start_time = datetime.now()
        k = k or self.top_k
        
        logger.info(f"Querying RAG system: '{question}' (k={k})")
        
        try:
            # Search for relevant documents
            search_results = self.vector_store.similarity_search(
                query=question,
                k=k,
                content_type_filter=content_type_filter
            )
            
            if not search_results:
                return RAGResponse(
                    answer="I couldn't find any relevant information for your question.",
                    sources=[],
                    query=question,
                    confidence=0.0,
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Prepare sources
            sources = []
            context_texts = []
            
            for text, distance, metadata in search_results:
                # Convert distance to similarity score (cosine distance -> similarity)
                similarity = 1.0 - distance
                
                source = {
                    "text": text,
                    "similarity": similarity,
                    "metadata": metadata
                }
                sources.append(source)
                context_texts.append(text)
            
            # Generate answer if LLM client is available
            answer = ""
            confidence = 0.0
            
            if generate_answer and self.llm_client:
                try:
                    context = "\n\n".join(context_texts)
                    answer, confidence = self._generate_answer(question, context)
                except Exception as e:
                    logger.error(f"Error generating answer: {e}")
                    answer = "I found relevant information but couldn't generate a comprehensive answer."
                    confidence = 0.5
            else:
                # Fallback: return relevant chunks as answer
                answer = f"Here are {len(search_results)} relevant pieces of information:\n\n"
                for i, (text, _, metadata) in enumerate(search_results, 1):
                    source_info = metadata.get("source_url", "Unknown source")
                    answer += f"{i}. From {source_info}:\n{text[:300]}{'...' if len(text) > 300 else ''}\n\n"
                confidence = max(1.0 - search_results[0][1], 0.0) if search_results else 0.0
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✓ Query processed in {processing_time:.2f}s, confidence: {confidence:.2f}")
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                query=question,
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(f"Query failed: {error_msg}")
            
            return RAGResponse(
                answer="I encountered an error while processing your question.",
                sources=[],
                query=question,
                confidence=0.0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                error=error_msg
            )
    
    def _generate_answer(self, question: str, context: str) -> Tuple[str, float]:
        """
        Generate an answer using the LLM client.
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Tuple of (answer, confidence)
        """
        prompt = f"""Based on the following context, please provide a comprehensive answer to the question.
If the context doesn't contain enough information to answer the question, please say so.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-3.5-turbo",
                max_tokens=1000,
                temperature=0.3
            )
            
            answer = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Calculate confidence based on context relevance and answer quality
            confidence = self._calculate_confidence(question, context, answer)
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def _calculate_confidence(self, question: str, context: str, answer: str) -> float:
        """
        Calculate confidence score for the generated answer.
        
        Args:
            question: Original question
            context: Retrieved context
            answer: Generated answer
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Simple heuristic-based confidence calculation
        # This could be made more sophisticated with embedding similarity, etc.
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence if answer is not too short or generic
        if len(answer) > 50 and "I don't know" not in answer.lower():
            confidence += 0.2
        
        # Increase confidence if context is substantial
        if len(context) > 200:
            confidence += 0.1
        
        # Increase confidence if answer mentions specific details from context
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(context_words.intersection(answer_words))
        
        if overlap > 10:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def get_document_status(self, source_url: str) -> Optional[DocumentStatus]:
        """Get processing status for a document."""
        return self.processing_status.get(source_url)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the vector store.
        
        Returns:
            List of document metadata
        """
        try:
            documents = self.vector_store.list_all_documents()
            
            # Group by source URL
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
            
            return list(doc_map.values())
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def delete_document(self, source_url: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            source_url: Source URL of document to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            deleted_count = self.vector_store.delete_by_source_url(source_url)
            
            # Remove from processing status
            if source_url in self.processing_status:
                del self.processing_status[source_url]
            
            logger.info(f"Deleted {deleted_count} chunks for document: {source_url}")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting document {source_url}: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary with system stats
        """
        try:
            vector_stats = self.vector_store.get_collection_stats()
            
            # Processing status summary
            status_summary = {
                "completed": 0,
                "processing": 0,
                "error": 0
            }
            
            for status in self.processing_status.values():
                status_summary[status.status] = status_summary.get(status.status, 0) + 1
            
            return {
                "vector_store": vector_stats,
                "processing_status": status_summary,
                "total_processed_documents": len(self.processing_status),
                "system_components": {
                    "extractor": "active",
                    "splitter": "active",
                    "vector_store": "active",
                    "llm_client": "active" if self.llm_client else "inactive"
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}
    
    def clear_all_documents(self) -> bool:
        """
        Clear all documents from the system.
        
        Returns:
            True if cleared successfully
        """
        try:
            self.vector_store.clear_collection()
            self.processing_status.clear()
            logger.info("Cleared all documents from RAG system")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            return False
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'extractor') and self.extractor.driver:
                self.extractor.driver.quit()
        except:
            pass
