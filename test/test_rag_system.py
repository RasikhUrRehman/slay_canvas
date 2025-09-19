"""
RAG System Test Suite
Comprehensive testing interface for the RAG (Retrieval-Augmented Generation) system.
Provides functions to check stats, add documents, and chat with the RAG system.
"""

import logging
import os
import sys
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import RAG system components
from engine.rag.rag_system import RAGSystem, RAGResponse, DocumentStatus
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_test.log')
    ]
)
logger = logging.getLogger(__name__)


class RAGTester:
    """Test interface for the RAG system with comprehensive functionality."""
    
    def __init__(self, collection_name: str = "test_rag_collection"):
        """
        Initialize the RAG tester.
        
        Args:
            collection_name: Name of the test collection
        """
        self.collection_name = collection_name
        self.rag_system = None
        self.test_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"🧪 RAG Tester initialized with session ID: {self.test_session_id}")
    
    def initialize_rag_system(self, chunk_size: int = 1000, chunk_overlap: int = 200, top_k: int = 5) -> bool:
        """
        Initialize the RAG system with custom parameters.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of top results to retrieve
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("🚀 Initializing RAG system...")
            self.rag_system = RAGSystem(
                collection_name=self.collection_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k
            )
            logger.info("✅ RAG system initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG system: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dict containing system stats or error info
        """
        if not self.rag_system:
            return {"error": "RAG system not initialized"}
        
        try:
            stats = self.rag_system.get_system_stats()
            
            # Add additional test-specific stats
            stats["test_session"] = {
                "session_id": self.test_session_id,
                "collection_name": self.collection_name,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("📊 System stats retrieved successfully")
            return stats
        except Exception as e:
            error_msg = f"Failed to get system stats: {e}"
            logger.error(f"❌ {error_msg}")
            return {"error": error_msg}
    
    def print_system_stats(self):
        """Print formatted system statistics to console."""
        stats = self.get_system_stats()
        
        if "error" in stats:
            print(f"❌ Error: {stats['error']}")
            return
        
        print("\n" + "="*60)
        print("📊 RAG SYSTEM STATISTICS")
        print("="*60)
        
        # Test session info
        if "test_session" in stats:
            session = stats["test_session"]
            print(f"🧪 Test Session ID: {session['session_id']}")
            print(f"📦 Collection Name: {session['collection_name']}")
            print(f"⏰ Timestamp: {session['timestamp']}")
            print("-" * 40)
        
        # Vector store stats
        if "vector_store" in stats:
            vs_stats = stats["vector_store"]
            print(f"📚 Total Documents: {vs_stats.get('total_documents', 'N/A')}")
            print(f"🔢 Total Chunks: {vs_stats.get('total_chunks', 'N/A')}")
            print(f"💾 Collection Size: {vs_stats.get('collection_size', 'N/A')}")
            print("-" * 40)
        
        # Processing stats
        if "processing" in stats:
            proc_stats = stats["processing"]
            print(f"✅ Successful Processes: {proc_stats.get('successful_documents', 0)}")
            print(f"❌ Failed Processes: {proc_stats.get('failed_documents', 0)}")
            print(f"⏱️  Average Processing Time: {proc_stats.get('average_processing_time', 0):.2f}s")
            print("-" * 40)
        
        # Recent documents
        if "recent_documents" in stats:
            recent = stats["recent_documents"]
            print(f"📄 Recent Documents ({len(recent)}):")
            for i, doc in enumerate(recent[:5], 1):
                status_emoji = "✅" if doc.get("status") == "completed" else "❌"
                print(f"  {i}. {status_emoji} {doc.get('source_url', 'Unknown')[:50]}...")
        
        print("="*60 + "\n")
    
    def add_document_from_url(self, url: str, show_progress: bool = True) -> DocumentStatus:
        """
        Add a document from URL with optional progress display.
        
        Args:
            url: URL of the document to process
            show_progress: Whether to show processing progress
            
        Returns:
            DocumentStatus object with processing results
        """
        if not self.rag_system:
            logger.error("❌ RAG system not initialized")
            return DocumentStatus(
                source_url=url,
                status="error",
                chunks_created=0,
                processing_time=0.0,
                error_message="RAG system not initialized"
            )
        
        try:
            logger.info(f"📥 Adding document from URL: {url}")
            
            if show_progress:
                print(f"🔄 Processing document: {url}")
                print("⏳ Extracting content...")
            
            start_time = time.time()
            status = self.rag_system.add_document_from_url(url)
            end_time = time.time()
            
            if show_progress:
                if status.status == "completed":
                    print(f"✅ Document processed successfully!")
                    print(f"📊 Created {status.chunks_created} chunks in {status.processing_time:.2f}s")
                else:
                    print(f"❌ Processing failed: {status.error_message}")
            
            return status
        except Exception as e:
            error_msg = f"Failed to add document: {e}"
            logger.error(f"❌ {error_msg}")
            return DocumentStatus(
                source_url=url,
                status="error",
                chunks_created=0,
                processing_time=0.0,
                error_message=error_msg
            )
    
    def add_custom_text(self, text: str, metadata: Dict[str, Any], show_progress: bool = True) -> DocumentStatus:
        """
        Add custom text with metadata.
        
        Args:
            text: Text content to add
            metadata: Metadata for the text
            show_progress: Whether to show processing progress
            
        Returns:
            DocumentStatus object with processing results
        """
        if not self.rag_system:
            logger.error("❌ RAG system not initialized")
            return DocumentStatus(
                source_url=metadata.get("source", "custom_text"),
                status="error",
                chunks_created=0,
                processing_time=0.0,
                error_message="RAG system not initialized"
            )
        
        try:
            if show_progress:
                print(f"📝 Adding custom text ({len(text)} characters)")
                print("⏳ Processing text...")
            
            status = self.rag_system.add_custom_text(text, metadata)
            
            if show_progress:
                if status.status == "completed":
                    print(f"✅ Text processed successfully!")
                    print(f"📊 Created {status.chunks_created} chunks in {status.processing_time:.2f}s")
                else:
                    print(f"❌ Processing failed: {status.error_message}")
            
            return status
        except Exception as e:
            error_msg = f"Failed to add custom text: {e}"
            logger.error(f"❌ {error_msg}")
            return DocumentStatus(
                source_url=metadata.get("source", "custom_text"),
                status="error",
                chunks_created=0,
                processing_time=0.0,
                error_message=error_msg
            )
    
    def chat_with_rag(self, question: str, k: Optional[int] = None, 
                     content_type_filter: Optional[str] = None,
                     generate_answer: bool = True, 
                     show_sources: bool = True) -> RAGResponse:
        """
        Chat with the RAG system using natural language queries.
        
        Args:
            question: Question to ask the RAG system
            k: Number of documents to retrieve (uses system default if None)
            content_type_filter: Filter by content type
            generate_answer: Whether to generate an answer using LLM
            show_sources: Whether to display source information
            
        Returns:
            RAGResponse object with answer and metadata
        """
        if not self.rag_system:
            logger.error("❌ RAG system not initialized")
            return RAGResponse(
                answer="Error: RAG system not initialized",
                sources=[],
                query=question,
                confidence=0.0,
                processing_time=0.0,
                error="RAG system not initialized"
            )
        
        try:
            logger.info(f"💬 Processing query: {question}")
            print(f"\n🤔 Question: {question}")
            print("🔍 Searching knowledge base...")
            
            response = self.rag_system.query(
                question=question,
                k=k,
                content_type_filter=content_type_filter,
                generate_answer=generate_answer
            )
            
            # Display results
            print(f"\n💡 Answer (Confidence: {response.confidence:.2f}):")
            print("-" * 50)
            print(response.answer)
            print("-" * 50)
            
            if show_sources and response.sources:
                print(f"\n📚 Sources ({len(response.sources)} found):")
                for i, source in enumerate(response.sources, 1):
                    print(f"\n{i}. 📄 {source.get('source_url', 'Unknown source')}")
                    print(f"   📊 Similarity: {source.get('similarity_score', 0):.3f}")
                    print(f"   📝 Content: {source.get('content', '')[:200]}...")
                    if source.get('metadata'):
                        content_type = source['metadata'].get('content_type', 'Unknown')
                        print(f"   🏷️  Type: {content_type}")
            
            print(f"\n⏱️  Processing time: {response.processing_time:.2f}s")
            
            return response
        except Exception as e:
            error_msg = f"Failed to process query: {e}"
            logger.error(f"❌ {error_msg}")
            return RAGResponse(
                answer=f"Error: {error_msg}",
                sources=[],
                query=question,
                confidence=0.0,
                processing_time=0.0,
                error=error_msg
            )
    
    def interactive_chat(self):
        """Start an interactive chat session with the RAG system."""
        if not self.rag_system:
            print("❌ RAG system not initialized. Please initialize first.")
            return
        
        print("\n" + "="*60)
        print("🤖 INTERACTIVE RAG CHAT SESSION")
        print("="*60)
        print("💡 Type your questions and press Enter")
        print("💡 Type 'stats' to see system statistics")
        print("💡 Type 'quit' or 'exit' to end the session")
        print("-" * 60)
        
        while True:
            try:
                question = input("\n🤔 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! Chat session ended.")
                    break
                
                if question.lower() == 'stats':
                    self.print_system_stats()
                    continue
                
                if not question:
                    print("❓ Please enter a question.")
                    continue
                
                # Process the question
                self.chat_with_rag(question)
                
            except KeyboardInterrupt:
                print("\n👋 Chat session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def run_test_suite(self, test_urls: List[str] = None, test_texts: List[Dict[str, Any]] = None):
        """
        Run a comprehensive test suite with sample data.
        
        Args:
            test_urls: List of URLs to test with
            test_texts: List of custom texts with metadata to test with
        """
        print("\n" + "="*60)
        print("🧪 RAG SYSTEM TEST SUITE")
        print("="*60)
        
        # Initialize system
        if not self.initialize_rag_system():
            print("❌ Failed to initialize RAG system. Aborting tests.")
            return
        
        # Test 1: System stats (empty system)
        print("\n1️⃣  Testing system stats (empty system)...")
        self.print_system_stats()
        
        # Test 2: Add sample documents
        if test_urls:
            print("\n2️⃣  Testing document addition from URLs...")
            for i, url in enumerate(test_urls, 1):
                print(f"\n📥 Adding document {i}/{len(test_urls)}: {url}")
                status = self.add_document_from_url(url)
                if status.status == "completed":
                    print(f"✅ Success: {status.chunks_created} chunks created")
                else:
                    print(f"❌ Failed: {status.error_message}")
        
        # Test 3: Add custom texts
        if test_texts:
            print("\n3️⃣  Testing custom text addition...")
            for i, text_data in enumerate(test_texts, 1):
                print(f"\n📝 Adding custom text {i}/{len(test_texts)}")
                status = self.add_custom_text(text_data["text"], text_data["metadata"])
                if status.status == "completed":
                    print(f"✅ Success: {status.chunks_created} chunks created")
                else:
                    print(f"❌ Failed: {status.error_message}")
        
        # Test 4: System stats (populated system)
        print("\n4️⃣  Testing system stats (populated system)...")
        self.print_system_stats()
        
        # Test 5: Sample queries
        sample_questions = [
            "What is the main topic of the documents?",
            "Can you summarize the key points?",
            "What are the most important concepts mentioned?"
        ]
        
        print("\n5️⃣  Testing RAG queries...")
        for i, question in enumerate(sample_questions, 1):
            print(f"\n🤔 Test query {i}/{len(sample_questions)}")
            response = self.chat_with_rag(question)
            if response.error:
                print(f"❌ Query failed: {response.error}")
            else:
                print(f"✅ Query successful (confidence: {response.confidence:.2f})")
        
        print("\n" + "="*60)
        print("🎉 TEST SUITE COMPLETED!")
        print("="*60)
    
    def cleanup(self):
        """Clean up test resources."""
        if self.rag_system:
            try:
                print("🧹 Cleaning up test resources...")
                self.rag_system.clear_all_documents()
                print("✅ Cleanup completed")
            except Exception as e:
                print(f"❌ Cleanup failed: {e}")


def main():
    """Main function to demonstrate RAG system testing."""
    # Create tester instance
    tester = RAGTester("demo_rag_test")
    
    # Sample test data
    test_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning"
    ]
    
    test_texts = [
        {
            "text": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. It has become an essential part of the technology industry. AI research has been highly successful in developing effective techniques for solving a wide range of problems, from game playing to medical diagnosis.",
            "metadata": {
                "source": "AI_overview",
                "content_type": "educational",
                "topic": "artificial_intelligence",
                "author": "Test Author"
            }
        },
        {
            "text": "Machine Learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.",
            "metadata": {
                "source": "ML_overview", 
                "content_type": "educational",
                "topic": "machine_learning",
                "author": "Test Author"
            }
        }
    ]
    
    try:
        # Run the test suite
        tester.run_test_suite(test_urls=test_urls, test_texts=test_texts)
        
        # Start interactive chat
        print("\n🚀 Starting interactive chat session...")
        tester.interactive_chat()
        
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        # Cleanup
        tester.cleanup()


if __name__ == "__main__":
    main()