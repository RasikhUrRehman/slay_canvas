"""
Test script for the Knowledge Base Agent
Tests the agent's ability to decide when to search and generate appropriate queries.
"""

import logging
import sys
import os
from typing import Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.llm.agent import KnowledgeBaseAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentTester:
    """Test suite for the Knowledge Base Agent"""
    
    def __init__(self):
        """Initialize the agent tester"""
        self.agent = None
    
    def initialize_agent(self) -> bool:
        """
        Initialize the Knowledge Base Agent.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("🚀 Initializing Knowledge Base Agent...")
            self.agent = KnowledgeBaseAgent(
                rag_collection_name="test_agent_kb",
                max_tokens=1000,
                temperature=0.3
            )
            logger.info("✅ Agent initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize agent: {e}")
            return False
    
    def test_knowledge_base_stats(self):
        """Test getting knowledge base statistics"""
        if not self.agent:
            print("❌ Agent not initialized")
            return
        
        print("\n📊 Testing knowledge base statistics...")
        stats = self.agent.get_knowledge_base_stats()
        print(f"Knowledge base stats: {stats}")
    
    def add_sample_documents(self):
        """Add some sample documents to test with"""
        if not self.agent:
            print("❌ Agent not initialized")
            return
        
        print("\n📄 Adding sample documents to knowledge base...")
        
        # Add some sample text documents
        sample_texts = [
            {
                "text": """
                Python is a high-level, interpreted programming language with dynamic semantics. 
                Its high-level built-in data structures, combined with dynamic typing and dynamic binding, 
                make it very attractive for Rapid Application Development, as well as for use as a scripting 
                or glue language to connect existing components together.
                """,
                "metadata": {"title": "Python Programming", "type": "documentation", "topic": "programming"}
            },
            {
                "text": """
                Machine Learning is a subset of artificial intelligence (AI) that provides systems 
                the ability to automatically learn and improve from experience without being explicitly programmed. 
                Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.
                """,
                "metadata": {"title": "Machine Learning Basics", "type": "documentation", "topic": "AI"}
            },
            {
                "text": """
                FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ 
                based on standard Python type hints. It's designed to be easy to use and learn, fast to code, 
                ready for production, and based on open standards.
                """,
                "metadata": {"title": "FastAPI Framework", "type": "documentation", "topic": "web development"}
            }
        ]
        
        for i, doc in enumerate(sample_texts, 1):
            print(f"Adding document {i}/{len(sample_texts)}: {doc['metadata']['title']}")
            result = self.agent.add_text_to_knowledge_base(doc["text"], doc["metadata"])
            if result["success"]:
                print(f"✅ Success: {result['chunks_created']} chunks created")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    def test_agent_queries(self):
        """Test the agent with various types of queries"""
        if not self.agent:
            print("❌ Agent not initialized")
            return
        
        print("\n🤖 Testing agent with various queries...")
        
        test_queries = [
            # Should trigger knowledge base search
            "What is Python programming?",
            "Tell me about machine learning",
            "Explain FastAPI framework",
            
            # Should not trigger knowledge base search
            "What's the weather like today?",
            "How are you doing?",
            "What's 2 + 2?",
            
            # Edge cases
            "Can you provide information about deep learning algorithms?",
            "Hello, how can you help me?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"🤔 Test Query {i}/{len(test_queries)}: {query}")
            print("-" * 60)
            
            try:
                response = self.agent.process_query(query)
                
                print(f"🧠 Reasoning: {response.reasoning}")
                print(f"🔧 Tools Used: {response.tools_used}")
                print(f"📊 Confidence: {response.confidence:.2f}")
                print(f"⏱️  Processing Time: {response.processing_time:.2f}s")
                
                if response.sources:
                    print(f"📚 Sources Found: {len(response.sources)}")
                    for j, source in enumerate(response.sources[:2], 1):  # Show first 2 sources
                        print(f"   Source {j}: {source['text'][:100]}...")
                
                print(f"💬 Answer: {response.answer}")
                
                if response.error:
                    print(f"❌ Error: {response.error}")
                
            except Exception as e:
                print(f"❌ Query failed: {e}")
    
    def test_knowledge_base_management(self):
        """Test knowledge base management functions"""
        if not self.agent:
            print("❌ Agent not initialized")
            return
        
        print("\n📋 Testing knowledge base management...")
        
        # List documents
        print("📄 Documents in knowledge base:")
        documents = self.agent.list_knowledge_base_documents()
        for i, doc in enumerate(documents, 1):
            print(f"   {i}. {doc.get('source', 'Unknown')} - {doc.get('chunks', 0)} chunks")
        
        # Get stats
        print("\n📊 Knowledge base statistics:")
        stats = self.agent.get_knowledge_base_stats()
        if "error" not in stats:
            print(f"   Total documents: {stats.get('total_documents', 0)}")
            print(f"   Total chunks: {stats.get('total_chunks', 0)}")
            print(f"   Collection name: {stats.get('collection_name', 'Unknown')}")
        else:
            print(f"   Error getting stats: {stats['error']}")
    
    def run_comprehensive_test(self):
        """Run a comprehensive test of the agent"""
        print("="*80)
        print("🧪 KNOWLEDGE BASE AGENT COMPREHENSIVE TEST")
        print("="*80)
        
        # Test 1: Initialize agent
        print("\n1️⃣  Testing agent initialization...")
        if not self.initialize_agent():
            print("❌ Agent initialization failed. Stopping tests.")
            return
        
        # Test 2: Check initial stats
        print("\n2️⃣  Testing initial knowledge base stats...")
        self.test_knowledge_base_stats()
        
        # Test 3: Add sample documents
        print("\n3️⃣  Testing document addition...")
        self.add_sample_documents()
        
        # Test 4: Test knowledge base management
        print("\n4️⃣  Testing knowledge base management...")
        self.test_knowledge_base_management()
        
        # Test 5: Test agent queries
        print("\n5️⃣  Testing agent query processing...")
        self.test_agent_queries()
        
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE TEST COMPLETED!")
        print("💡 The agent has been tested with:")
        print("   - Knowledge base initialization")
        print("   - Document addition and management")
        print("   - Query processing with tool decision making")
        print("   - Automatic search query generation")
        print("="*80)


def main():
    """Main function to run the agent tests"""
    tester = AgentTester()
    
    try:
        tester.run_comprehensive_test()
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user. Goodbye!")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()