"""
Vector store implementation for RAG-based system using Milvus and custom NLP Cloud embeddings.
"""

import logging
import os
import uuid
from typing import Dict, List, Optional, Tuple, Any

from pymilvus import (
    Collection, CollectionSchema, DataType, FieldSchema,
    connections, utility, MilvusClient
)
from dotenv import load_dotenv

# Import the custom embedding service
from engine.services.embedding import EmbeddingService
from app.core.config import settings

# Load environment variables from .env.local
load_dotenv(".env.local")
# Also load main .env file for Docker compatibility
load_dotenv()

logger = logging.getLogger("uvicorn")


class VectorStore:
    """Vector store for document embeddings using Milvus and NLP Cloud embeddings."""

    def __init__(self, collection_name: str = "documents", dimension: int = 1536):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection to store documents
            dimension: Dimension of the embeddings (1536 for openai/text-embedding-3-small)
        """
        self.collection_name = collection_name
        self.dimension = dimension

        # Initialize embedding service
        self.embedding_service = EmbeddingService()

        # Connect to Milvus server using settings
        milvus_host = settings.MILVUS_HOST
        milvus_port = settings.MILVUS_PORT
        
        self.milvus_uri = f"http://{milvus_host}:{milvus_port}"
        logger.info(f"Connecting to Milvus server at {self.milvus_uri}")

        self.client = MilvusClient(uri=self.milvus_uri)
        
        # Check if collection exists, create if not
        if not self.client.has_collection(collection_name=collection_name):
            self._create_collection()
        else:
            # Load existing collection
            self._ensure_collection_loaded()
        
        logger.info(f"Connected to Milvus with collection: {collection_name}")
        logger.info(f"Using Hugging Face embedding model: {self.embedding_service.model}")

    def _create_collection(self):
        """Create the Milvus collection with appropriate schema."""
        # Define collection schema
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True
        )
        
        # Add fields
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
        
        # Metadata fields
        schema.add_field(field_name="source_url", datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
        schema.add_field(field_name="total_chunks", datatype=DataType.INT64)
        schema.add_field(field_name="chunk_size", datatype=DataType.INT64)
        schema.add_field(field_name="extraction_time", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="transcription_type", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="original_filename", datatype=DataType.VARCHAR, max_length=500)
        # User and project identification fields
        schema.add_field(field_name="user_id", datatype=DataType.INT64)
        schema.add_field(field_name="project_name", datatype=DataType.VARCHAR, max_length=200)
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema
        )

        # Create index for vector field
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            params={"nlist": 1024}
        )
        
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )
        
        # Load the collection into memory so it can be searched
        self.client.load_collection(collection_name=self.collection_name)
        
        logger.info(f"Created and loaded collection: {self.collection_name}")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using the embedding service.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []
            for text in texts:
                result = self.embedding_service.get_embedding(text)
                if result and 'embeddings' in result:
                    embeddings.append(result['embeddings'])
                else:
                    logger.error(f"No embeddings returned from embedding service: {result}")
                    raise ValueError("Failed to get embeddings from embedding service")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text chunks to add
            metadatas: Optional metadata for each text chunk
            
        Returns:
            List of document IDs
        """
        if not texts:
            return []

        # Generate embeddings using the embedding service
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self._get_embeddings(texts)
        logger.info(f"Generated {len(embeddings)} embeddings")

        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in texts]

        # Prepare data for insertion
        entities = []
        for i, (text, embedding, doc_id) in enumerate(zip(texts, embeddings, ids)):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            
            entity = {
                "id": doc_id,
                "text": text,
                "vector": embedding,
                "source_url": metadata.get("source_url", ""),
                "content_type": metadata.get("content_type", ""),
                "title": metadata.get("title", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "total_chunks": metadata.get("total_chunks", 1),
                "chunk_size": metadata.get("chunk_size", len(text)),
                "extraction_time": metadata.get("extraction_time", ""),
                "transcription_type": metadata.get("transcription_type", "text"),
                "original_filename": metadata.get("original_filename", ""),
                "user_id": metadata.get("user_id", 0),
                "project_name": metadata.get("project_name", "")
            }
            entities.append(entity)

        logger.info(f"Prepared {len(entities)} entities for insertion")

        # Insert data
        insert_result = self.client.insert(
            collection_name=self.collection_name,
            data=entities
        )
        logger.info(f"Insert completed. Result: {insert_result}")
        
        # Flush to ensure data is written to disk
        flush_result = self.client.flush(collection_name=self.collection_name)
        logger.info(f"Flush completed. Result: {flush_result}")
        
        # Ensure collection is loaded after inserting data
        self._ensure_collection_loaded()

        logger.info(f"Added {len(texts)} documents to vector store")
        return ids

    def _ensure_collection_loaded(self):
        """Ensure the collection is loaded into memory for searching."""
        try:
            # Always try to load the collection to be safe
            self.client.load_collection(collection_name=self.collection_name)
            logger.info(f"Collection {self.collection_name} loaded successfully")
        except Exception as e:
            # If loading fails, check if it's already loaded
            try:
                load_state = self.client.get_load_state(collection_name=self.collection_name)
                if load_state["state"] == "Loaded":
                    logger.info(f"Collection {self.collection_name} is already loaded")
                else:
                    logger.error(f"Collection {self.collection_name} is not loaded. State: {load_state['state']}")
                    raise Exception(f"Collection not loaded: {load_state['state']}")
            except Exception as check_error:
                logger.error(f"Failed to check/load collection: {e}, check error: {check_error}")
                raise

    def similarity_search(self, query: str, k: int = 5, content_type_filter: Optional[str] = None, 
                         user_id: Optional[int] = None, project_name: Optional[str] = None) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            content_type_filter: Optional filter by content type
            user_id: Optional filter by user ID
            project_name: Optional filter by project name
            
        Returns:
            List of tuples (document, distance, metadata)
        """
        # Ensure collection is loaded
        self._ensure_collection_loaded()
        
        # Generate query embedding using the embedding service
        query_embedding = self._get_embeddings([query])[0]

        # Build filter expression
        filter_conditions = []
        if content_type_filter:
            filter_conditions.append(f'content_type == "{content_type_filter}"')
        if user_id is not None:
            filter_conditions.append(f'user_id == {user_id}')
        if project_name:
            filter_conditions.append(f'project_name == "{project_name}"')
        
        filter_expr = " && ".join(filter_conditions) if filter_conditions else None

        # Search in collection
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="vector",
            search_params=search_params,
            limit=k,
            filter=filter_expr,
            output_fields=["text", "source_url", "content_type", "title", "chunk_index", 
                          "total_chunks", "chunk_size", "extraction_time", "transcription_type", "original_filename",
                          "user_id", "project_name"]
        )

        # Format results
        formatted_results = []
        for hit in results[0]:
            metadata = {
                "source_url": hit["entity"]["source_url"],
                "content_type": hit["entity"]["content_type"],
                "title": hit["entity"]["title"],
                "chunk_index": hit["entity"]["chunk_index"],
                "total_chunks": hit["entity"]["total_chunks"],
                "chunk_size": hit["entity"]["chunk_size"],
                "extraction_time": hit["entity"]["extraction_time"],
                "transcription_type": hit["entity"]["transcription_type"],
                "original_filename": hit["entity"].get("original_filename", ""),
                "user_id": hit["entity"].get("user_id", 0),
                "project_name": hit["entity"].get("project_name", "")
            }
            formatted_results.append((hit["entity"]["text"], hit["distance"], metadata))

        return formatted_results

    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        self._ensure_collection_loaded()
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        return stats["row_count"]

    def clear_collection(self):
        """Clear all documents from the collection."""
        self.client.drop_collection(collection_name=self.collection_name)
        self._create_collection()
        logger.info(f"Cleared collection: {self.collection_name}")

    def drop_collection(self):
        """Completely drop the collection without recreating it."""
        self.client.drop_collection(collection_name=self.collection_name)
        logger.info(f"Dropped collection: {self.collection_name}")

    def search_by_metadata(self, source_url: str = None, content_type: str = None, k: int = 10) -> List[Tuple[str, Dict]]:
        """
        Search documents by metadata filters.
        
        Args:
            source_url: Filter by source URL
            content_type: Filter by content type
            k: Number of results to return
            
        Returns:
            List of (text, metadata) tuples
        """
        try:
            # Ensure collection is loaded
            self._ensure_collection_loaded()
            
            # Build filter expression
            filter_expr = []
            
            if source_url:
                filter_expr.append(f'source_url == "{source_url}"')
            
            if content_type:
                filter_expr.append(f'content_type == "{content_type}"')
            
            # Join with AND if multiple filters
            filter_str = " && ".join(filter_expr) if filter_expr else None
            
            # Search with filters
            results = self.client.search(
                collection_name=self.collection_name,
                data=[[0.0] * self.dimension],  # Dummy embedding for metadata-only search
                filter=filter_str,
                limit=k,
                output_fields=["text", "source_url", "content_type", "title", "chunk_index", 
                              "total_chunks", "chunk_size", "extraction_time", "transcription_type"]
            )
            
            # Convert results to expected format
            documents = []
            if results and len(results) > 0:
                for hit in results[0]:
                    entity = hit['entity']
                    metadata = {
                        'source_url': entity.get('source_url', ''),
                        'content_type': entity.get('content_type', ''),
                        'title': entity.get('title', ''),
                        'chunk_index': entity.get('chunk_index', 0),
                        'total_chunks': entity.get('total_chunks', 0),
                        'chunk_size': entity.get('chunk_size', 0),
                        'extraction_time': entity.get('extraction_time', ''),
                        'transcription_type': entity.get('transcription_type', '')
                    }
                    documents.append((entity.get('text', ''), metadata))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []

    def delete_by_source_url(self, source_url: str) -> int:
        """
        Delete all documents with the specified source URL.
        
        Args:
            source_url: The source URL to delete
            
        Returns:
            Number of documents deleted
        """
        try:
            # Ensure collection is loaded
            self._ensure_collection_loaded()
            
            # First, find all IDs for this source URL
            filter_expr = f'source_url == "{source_url}"'
            
            # Query to get all IDs for this source URL
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["id"]
            )

            logger.info(f"Found {len(results) if results else 0} documents with source URL: {source_url}")
            
            if not results:
                logger.info(f"No documents found with source URL: {source_url}")
                return 0
            
            # Extract IDs
            ids_to_delete = [result['id'] for result in results]
            
            # Delete the documents
            delete_result = self.client.delete(
                collection_name=self.collection_name,
                filter=filter_expr
            )

            logger.info(f"Delete result: {delete_result}")

            self.client.flush(self.collection_name)
            
            self.client.release_collection(self.collection_name)

            # Load collection (reload into memory)
            self.client.load_collection(self.collection_name)

            logger.info(f"Deleted {len(ids_to_delete)} documents with source URL: {source_url}")
            return len(ids_to_delete)
            
        except Exception as e:
            logger.error(f"Error deleting documents with source URL {source_url}: {e}")
            return 0

    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            # Get collection info
            collection_info = self.client.describe_collection(self.collection_name)
            
            # Count total entities
            stats = self.client.get_collection_stats(self.collection_name)
            
            return {
                'collection_name': self.collection_name,
                'total_entities': stats['row_count'],
                'dimension': self.dimension,
                'collection_info': collection_info
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def list_all_documents(self) -> List[Tuple[str, Dict]]:
        """
        List all documents in the collection.
        
        Returns:
            List of (text, metadata) tuples for all documents
        """
        try:
            # Query all documents
            results = self.client.query(
                collection_name=self.collection_name,
                filter="",  # No filter to get all
                output_fields=["text", "source_url", "content_type", "title", "chunk_index", 
                              "total_chunks", "chunk_size", "extraction_time", "transcription_type", "original_filename"],
                limit=10000  # Large limit to get all documents
            )
            
            documents = []
            if results:
                for result in results:
                    metadata = {
                        'source_url': result.get('source_url', ''),
                        'content_type': result.get('content_type', ''),
                        'title': result.get('title', ''),
                        'chunk_index': result.get('chunk_index', 0),
                        'total_chunks': result.get('total_chunks', 0),
                        'chunk_size': result.get('chunk_size', 0),
                        'extraction_time': result.get('extraction_time', ''),
                        'transcription_type': result.get('transcription_type', ''),
                        'original_filename': result.get('original_filename', '')
                    }
                    documents.append((result.get('text', ''), metadata))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing all documents: {e}")
            return []
