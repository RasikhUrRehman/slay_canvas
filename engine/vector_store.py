"""
Vector store implementation for RAG-based voice agent using Milvus and OpenAI embeddings.
"""

import logging
import os
import uuid
from typing import Dict, List, Optional, Tuple

from pymilvus import (
    Collection, CollectionSchema, DataType, FieldSchema,
    connections, utility, MilvusClient
)
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env.local
load_dotenv(".env.local")

logger = logging.getLogger("uvicorn")


class VectorStore:
    """Vector store for document embeddings using Milvus and OpenAI embeddings."""

    def __init__(self, collection_name: str = "documents",
                 embedding_model: str = "text-embedding-3-small", dimension: int = 1536):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection to store documents
            embedding_model: OpenAI embedding model to use (e.g., text-embedding-3-small, text-embedding-3-large)
            dimension: Dimension of the embeddings (1536 for text-embedding-3-small)
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.dimension = dimension

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.openai_client = OpenAI(api_key=api_key)

        # Connect to Milvus server
        milvus_host = os.getenv("MILVUS_HOST", "milvus")
        milvus_port = os.getenv("MILVUS_PORT", "19531")
        
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
        logger.info(f"Using OpenAI embedding model: {embedding_model}")

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
        schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="chapter", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
        schema.add_field(field_name="total_chunks", datatype=DataType.INT64)
        schema.add_field(field_name="chunk_size", datatype=DataType.INT64)
        
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
        Get embeddings for a list of texts using OpenAI API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
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

        # Generate embeddings using OpenAI
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
                "filename": metadata.get("filename", ""),
                "chapter": metadata.get("chapter", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "total_chunks": metadata.get("total_chunks", 1),
                "chunk_size": metadata.get("chunk_size", len(text))
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

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of tuples (document, distance, metadata)
        """
        # Ensure collection is loaded
        self._ensure_collection_loaded()
        
        # Generate query embedding using OpenAI
        query_embedding = self._get_embeddings([query])[0]

        # Search in collection
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="vector",
            search_params=search_params,
            limit=k,
            output_fields=["text", "filename", "chapter", "chunk_index", "total_chunks", "chunk_size"]
        )

        # Format results
        formatted_results = []
        for hit in results[0]:
            metadata = {
                "filename": hit["entity"]["filename"],
                "chapter": hit["entity"]["chapter"],
                "chunk_index": hit["entity"]["chunk_index"],
                "total_chunks": hit["entity"]["total_chunks"],
                "chunk_size": hit["entity"]["chunk_size"]
            }
            formatted_results.append((hit["entity"]["text"], hit["distance"], metadata))

        return formatted_results
    def semantic_search(self, query: str, k: int = 5, expand_query: bool = False) -> List[Tuple[str, float, Dict]]:
        """
        Perform semantic search with query expansion and context understanding.
        
        Args:
            query: Query text
            k: Number of results to return
            expand_query: Whether to expand the query with semantic variations
            
        Returns:
            List of tuples (document, distance, metadata)
        """
        # Ensure collection is loaded
        self._ensure_collection_loaded()
        
        # Expand query for better semantic understanding
        expanded_query = query
        if expand_query:
            expanded_query = self._expand_query_semantically(query)

        logger.info(f"Original query: {query}")
        if expanded_query != query:
            logger.info(f"Expanded query: {expanded_query}")

        # Generate query embedding using OpenAI
        query_embedding = self._get_embeddings([expanded_query])[0]

        # Search in collection
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="vector",
            search_params=search_params,
            limit=k * 2,  # Get more results for reranking
            output_fields=["text", "filename", "chapter", "chunk_index", "total_chunks", "chunk_size"]
        )

        # Format results
        formatted_results = []
        for hit in results[0]:
            metadata = {
                "filename": hit["entity"]["filename"],
                "chapter": hit["entity"]["chapter"],
                "chunk_index": hit["entity"]["chunk_index"],
                "total_chunks": hit["entity"]["total_chunks"],
                "chunk_size": hit["entity"]["chunk_size"]
            }
            formatted_results.append((hit["entity"]["text"], hit["distance"], metadata))

        # Apply semantic reranking
        reranked_results = self._semantic_rerank(query, formatted_results)

        return reranked_results[:k]

    def _expand_query_semantically(self, query: str) -> str:
        """
        Expand query with semantic variations using OpenAI.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query with semantic variations
        """
        try:
            expansion_prompt = f"""
Given this search query, expand it with semantic variations and related terms to improve search results:

Query: "{query}"

Provide an expanded version that includes:
- Synonyms and related terms
- Different ways to phrase the same concept
- Context that would help find relevant documents

Expanded query:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": expansion_prompt}],
                max_tokens=100,
                temperature=0.3
            )

            expanded = response.choices[0].message.content.strip()
            return expanded if expanded else query

        except Exception as e:
            logger.warning(f"Query expansion failed, using original query: {e}")
            return query

    def _semantic_rerank(self, original_query: str, results: List[Tuple[str, float, Dict]]) -> List[Tuple[str, float, Dict]]:
        """
        Rerank results based on semantic relevance.
        
        Args:
            original_query: Original user query
            results: List of (document, distance, metadata) tuples
            
        Returns:
            Reranked results
        """
        if not results:
            return results

        try:
            # Create a prompt for semantic scoring
            documents = [doc for doc, _, _ in results]

            scoring_prompt = f"""
Rate the semantic relevance of each document to the query on a scale of 1-10:

Query: "{original_query}"

Documents:
"""
            for i, doc in enumerate(documents):
                scoring_prompt += f"{i+1}. {doc[:200]}{'...' if len(doc) > 200 else ''}\n"

            scoring_prompt += "\nProvide scores as: 1:score, 2:score, 3:score, etc."

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": scoring_prompt}],
                max_tokens=100,
                temperature=0.1
            )

            # Parse scores
            scores_text = response.choices[0].message.content.strip()
            scores = self._parse_relevance_scores(scores_text, len(documents))

            # Combine distance and semantic scores
            reranked = []
            for i, (doc, distance, metadata) in enumerate(results):
                semantic_score = scores.get(i, 5.0)  # Default score if parsing fails
                # Combine distance (lower is better) with semantic score (higher is better)
                # Normalize and combine: lower combined_score is better
                combined_score = distance - (semantic_score / 10.0)
                reranked.append((doc, combined_score, metadata))

            # Sort by combined score (lower is better)
            reranked.sort(key=lambda x: x[1])

            return reranked

        except Exception as e:
            logger.warning(f"Semantic reranking failed, using original order: {e}")
            return results

    def _parse_relevance_scores(self, scores_text: str, num_docs: int) -> Dict[int, float]:
        """Parse relevance scores from GPT response."""
        scores = {}
        try:
            for line in scores_text.split('\n'):
                if ':' in line:
                    parts = line.strip().split(':')
                    if len(parts) >= 2:
                        doc_idx = int(parts[0]) - 1  # Convert to 0-based index
                        score = float(parts[1])
                        if 0 <= doc_idx < num_docs:
                            scores[doc_idx] = score
        except:
            pass
        return scores

    def hybrid_search(self, query: str, k: int = 5, similarity_weight: float = 0.7) -> List[Tuple[str, float, Dict]]:
        """
        Perform hybrid search combining similarity and semantic search.
        
        Args:
            query: Query text
            k: Number of results to return
            similarity_weight: Weight for similarity search (0.0-1.0) 
        Returns:
            List of tuples (document, combined_score, metadata)
        """
        # Get results from both methods
        similarity_results = self.similarity_search(query, k * 2)
        semantic_results = self.semantic_search(query, k * 2, expand_query=True)

        # Combine results with weighting
        doc_scores = {}

        # Process similarity results
        for i, (doc, distance, metadata) in enumerate(similarity_results):
            # Convert distance to score (lower distance = higher score)
            similarity_score = 1.0 / (1.0 + distance)
            doc_scores[doc] = {
                'similarity': similarity_score,
                'semantic': 0.0,
                'metadata': metadata,
                'position': i
            }

        # Process semantic results
        for i, (doc, score, metadata) in enumerate(semantic_results):
            if doc in doc_scores:
                # Convert semantic score to 0-1 range (assuming it's already normalized)
                semantic_score = 1.0 / (1.0 + abs(score))
                doc_scores[doc]['semantic'] = semantic_score
            else:
                semantic_score = 1.0 / (1.0 + abs(score))
                doc_scores[doc] = {
                    'similarity': 0.0,
                    'semantic': semantic_score,
                    'metadata': metadata,
                    'position': i + len(similarity_results)
                }

        # Calculate combined scores
        results = []
        for doc, scores in doc_scores.items():
            combined_score = (
                similarity_weight * scores['similarity'] +
                (1 - similarity_weight) * scores['semantic']
            )
            results.append((doc, combined_score, scores['metadata']))

        # Sort by combined score (higher is better)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:k]

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

    def search_by_metadata(self, filename: str = None, chapter: str = None, k: int = 10) -> List[Tuple[str, Dict]]:
        """
        Search documents by metadata filters.
        
        Args:
            filename: Filter by filename
            chapter: Filter by chapter
            k: Number of results to return
            
        Returns:
            List of (text, metadata) tuples
        """
        try:
            # Ensure collection is loaded
            self._ensure_collection_loaded()
            
            # Build filter expression
            filter_expr = []
            
            if filename:
                filter_expr.append(f'filename == "{filename}"')
            
            if chapter:
                filter_expr.append(f'chapter == "{chapter}"')
            
            # Join with AND if multiple filters
            filter_str = " && ".join(filter_expr) if filter_expr else None
            
            # Search with filters
            results = self.client.search(
                collection_name=self.collection_name,
                data=[[0.0] * self.dimension],  # Dummy embedding for metadata-only search
                filter=filter_str,
                limit=k,
                output_fields=["text", "filename", "chapter", "chunk_index", "total_chunks", "chunk_size"]
            )
            
            # Convert results to expected format
            documents = []
            if results and len(results) > 0:
                for hit in results[0]:
                    entity = hit['entity']
                    metadata = {
                        'filename': entity.get('filename', ''),
                        'chapter': entity.get('chapter', ''),
                        'chunk_index': entity.get('chunk_index', 0),
                        'total_chunks': entity.get('total_chunks', 0),
                        'chunk_size': entity.get('chunk_size', 0)
                    }
                    documents.append((entity.get('text', ''), metadata))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []

    def delete_by_filename(self, filename: str) -> int:
        """
        Delete all documents with the specified filename.
        
        Args:
            filename: The filename to delete
            
        Returns:
            Number of documents deleted
        """
        try:
            # Ensure collection is loaded
            self._ensure_collection_loaded()
            
            # First, find all IDs for this filename
            filter_expr = f'filename == "{filename}"'
            
            # Query to get all IDs for this filename
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["id"]
            )

            logger.info(f"Found {len(results) if results else 0} documents with filename: {filename}")
            
            if not results:
                logger.info(f"No documents found with filename: {filename}")
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

            logger.info(f"Deleted {len(ids_to_delete)} documents with filename: {filename}")
            return len(ids_to_delete)
            
        except Exception as e:
            logger.error(f"Error deleting documents with filename {filename}: {e}")
            return 0

    def delete_by_metadata(self, filename: str = None, chapter: str = None) -> int:
        """
        Delete documents by metadata filters.
        
        Args:
            filename: Filter by filename
            chapter: Filter by chapter
            
        Returns:
            Number of documents deleted
        """
        try:
            # Build filter expression
            filter_expr = []
            
            if filename:
                filter_expr.append(f'filename == "{filename}"')
            
            if chapter:
                filter_expr.append(f'chapter == "{chapter}"')
            
            if not filter_expr:
                logger.warning("No filters provided for deletion")
                return 0
            
            # Join with AND if multiple filters
            filter_str = " && ".join(filter_expr)
            
            # First, count how many documents match
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_str,
                output_fields=["id"]
            )
            
            if not results:
                logger.info(f"No documents found matching filters: {filter_str}")
                return 0
            
            # Delete the documents
            delete_result = self.client.delete(
                collection_name=self.collection_name,
                filter=filter_str
            )
            
            count = len(results)
            logger.info(f"Deleted {count} documents matching filters: {filter_str}")
            return count
            
        except Exception as e:
            logger.error(f"Error deleting documents by metadata: {e}")
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
                output_fields=["text", "filename", "chapter", "chunk_index", "total_chunks", "chunk_size"],
                limit=10000  # Large limit to get all documents
            )
            
            documents = []
            if results:
                for result in results:
                    metadata = {
                        'filename': result.get('filename', ''),
                        'chapter': result.get('chapter', ''),
                        'chunk_index': result.get('chunk_index', 0),
                        'total_chunks': result.get('total_chunks', 0),
                        'chunk_size': result.get('chunk_size', 0)
                    }
                    documents.append((result.get('text', ''), metadata))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing all documents: {e}")
            return []