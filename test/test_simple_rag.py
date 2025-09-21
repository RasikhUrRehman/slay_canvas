"""
Simple RAG System Test
A minimal test to debug connection issues and verify basic functionality.
"""

import logging
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_settings():
    """Test if settings are loaded correctly."""
    print("🔧 Testing Settings Configuration:")
    print(f"   MILVUS_HOST: {settings.MILVUS_HOST}")
    print(f"   MILVUS_PORT: {settings.MILVUS_PORT}")
    print(f"   NLPCLOUD_TOKEN: {'***' if settings.NLPCLOUD_TOKEN else 'Not set'}")
    print(f"   OPENROUTER_API_KEY: {'***' if settings.OPENROUTER_API_KEY else 'Not set'}")
    print()

def test_milvus_connection():
    """Test direct Milvus connection."""
    try:
        from pymilvus import MilvusClient
        
        milvus_uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
        print(f"🔌 Testing Milvus connection to: {milvus_uri}")
        
        client = MilvusClient(uri=milvus_uri)
        
        # Test basic operations
        collections = client.list_collections()
        print(f"✅ Connected successfully! Collections: {collections}")
        
        return True
    except Exception as e:
        print(f"❌ Milvus connection failed: {e}")
        return False

def test_embedding_service():
    """Test the embedding service."""
    try:
        from engine.services.embedding import EmbeddingService
        
        print("🧠 Testing Embedding Service...")
        embedding_service = EmbeddingService()
        
        # Test embedding generation
        test_texts = ["Hello world", "This is a test"]
        embeddings = embedding_service.get_embeddings(test_texts)
        
        print(f"✅ Embeddings generated successfully!")
        print(f"   Model: {embedding_service.model}")
        print(f"   Embedding dimension: {len(embeddings[0]) if embeddings else 'N/A'}")
        print(f"   Number of embeddings: {len(embeddings)}")
        
        return True
    except Exception as e:
        print(f"❌ Embedding service failed: {e}")
        return False

def test_document_splitter():
    """Test the document splitter."""
    try:
        from engine.rag.document_splitter import DocumentSplitter
        
        print("📄 Testing Document Splitter...")
        splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)
        
        # Test text splitting
        test_text = "This is a long text that should be split into multiple chunks. " * 10
        test_metadata = {"source": "test", "content_type": "text"}
        
        chunks = splitter.split_custom_text(test_text, test_metadata)
        
        print(f"✅ Document splitter working!")
        print(f"   Original text length: {len(test_text)}")
        print(f"   Number of chunks: {len(chunks)}")
        print(f"   First chunk length: {len(chunks[0].text) if chunks else 'N/A'}")
        
        return True
    except Exception as e:
        print(f"❌ Document splitter failed: {e}")
        return False

def test_vector_store():
    """Test the vector store with minimal operations."""
    try:
        from engine.rag.vector_store import VectorStore
        
        print("🗄️  Testing Vector Store...")
        vector_store = VectorStore(collection_name="test_simple_collection")
        
        # Test adding a simple document
        test_texts = ["This is a test document for the vector store."]
        test_metadata = [{"source": "test", "content_type": "text"}]
        
        doc_ids = vector_store.add_documents(test_texts, test_metadata)
        print(f"✅ Vector store working!")
        print(f"   Document IDs: {doc_ids}")
        
        # Test search
        results = vector_store.similarity_search("test document", k=1)
        print(f"   Search results: {len(results)} found")
        
        # Cleanup
        vector_store.clear_collection()
        print("   Cleanup completed")
        
        return True
    except Exception as e:
        print(f"❌ Vector store failed: {e}")
        return False

def test_rag_system_minimal():
    """Test minimal RAG system functionality."""
    try:
        from engine.rag.rag_system import RAGSystem
        
        print("🤖 Testing RAG System (minimal)...")
        rag_system = RAGSystem(collection_name="test_minimal_rag")
        
        # Add a simple text
        test_metadata = {
            "source": "test_doc",
            "content_type": "text",
            "title": "Test Document"
        }
        
        status = rag_system.add_custom_text(
            "Artificial intelligence is a fascinating field of computer science.", 
            test_metadata
        )
        
        print(f"✅ RAG system working!")
        print(f"   Document status: {status.status}")
        print(f"   Chunks created: {status.chunks_created}")
        
        # Test query (without LLM generation to avoid API key issues)
        response = rag_system.query("What is artificial intelligence?", generate_answer=False)
        print(f"   Query results: {len(response.sources)} sources found")
        
        # Cleanup
        rag_system.clear_all_documents()
        print("   Cleanup completed")
        
        return True
    except Exception as e:
        print(f"❌ RAG system failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 SIMPLE RAG SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        ("Settings Configuration", test_settings),
        ("Milvus Connection", test_milvus_connection),
        ("Embedding Service", test_embedding_service),
        ("Document Splitter", test_document_splitter),
        ("Vector Store", test_vector_store),
        ("RAG System (Minimal)", test_rag_system_minimal),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 40)
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RAG system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()