"""
Test script to verify source attribution fixes in the RAG system.
This script creates a simple knowledge base and tests if real filenames are shown.
"""

import logging
import tempfile
import os
from engine.rag.rag_system import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_source_attribution():
    """Test that source attribution shows real filenames instead of 'unknown'."""
    
    print("🧪 Testing Source Attribution Fixes")
    print("=" * 50)
    
    try:
        # Initialize RAG system with a test collection
        rag_system = RAGSystem(collection_name="test_source_attribution")
        print("✓ RAG system initialized")
        
        # Create a temporary file with known content
        test_content = """
        Customer Support FAQ
        
        1. Account & Login
        How do I create an account?
        Go to the sign-up page, enter your email address, choose a password, and verify your email.
        
        I forgot my password. What should I do?
        Click on "Forgot Password" on the login page and follow the instructions sent to your email.
        
        2. Technical Support
        The app isn't working. What should I do?
        Try clearing your app cache and restarting your device. If the problem continues, reinstall the app.
        
        3. Delivery Information
        My package is delayed. What should I do?
        Please wait 24 hours after the expected delivery date. If it hasn't arrived, contact customer support.
        """
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, 
                                       prefix='customer_support_faq_') as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name
            
        # Extract the original filename from the temp file path
        original_filename = "Customer_Support_FAQ.txt"
        
        print(f"📄 Created test file: {original_filename}")
        print(f"   Temp path: {temp_file_path}")
        
        # Add the document using the RAG system
        print("\n📥 Adding document to RAG system...")
        status = rag_system.add_document_from_file(
            file_path=temp_file_path,
            original_filename=original_filename
        )
        
        if status.status == "completed":
            print(f"✓ Document added successfully - {status.chunks_created} chunks created")
        else:
            print(f"❌ Failed to add document: {status.error_message}")
            return False
            
        # Test query to check source attribution
        print("\n🔍 Testing query with source attribution...")
        query_result = rag_system.query("How do I create an account?")
        
        print(f"\nQuery: 'How do I create an account?'")
        print(f"Answer: {query_result.answer[:200]}...")
        print(f"Confidence: {query_result.confidence}")
        print(f"Sources found: {len(query_result.sources)}")
        
        # Check if sources show the real filename
        success = True
        for i, source in enumerate(query_result.sources, 1):
            print(f"\nSource {i}:")
            print(f"  Text: {source['text'][:100]}...")
            print(f"  Similarity: {source['similarity']:.4f}")
            print(f"  Source: {source.get('source', 'NOT FOUND')}")
            print(f"  Source Name: {source.get('source_name', 'NOT FOUND')}")
            
            # Check if the source shows the real filename
            source_name = source.get('source_name', '')
            if source_name == 'Unknown' or source_name == '' or 'tmp' in source_name.lower():
                print(f"  ❌ Source attribution failed - showing: '{source_name}'")
                success = False
            else:
                print(f"  ✓ Source attribution working - showing: '{source_name}'")
        
        # Test document listing
        print("\n📋 Testing document listing...")
        documents = rag_system.list_documents()
        
        if isinstance(documents, dict):
            for doc_name, doc_info in documents.items():
                print(f"\nDocument: {doc_name}")
                print(f"  Title: {doc_info.get('title', 'N/A')}")
                print(f"  Chunks: {doc_info.get('chunk_count', 0)}")
                print(f"  Characters: {doc_info.get('total_characters', 0)}")
                
                # Check if document name shows real filename
                if doc_name == original_filename or original_filename in doc_name:
                    print(f"  ✓ Document listing shows real filename")
                else:
                    print(f"  ❌ Document listing shows wrong name: '{doc_name}'")
                    success = False
        else:
            print(f"  Documents returned as list with {len(documents)} items")
            for i, doc in enumerate(documents):
                print(f"  Document {i+1}: {doc}")
                # For now, just check if we have documents
                if documents:
                    print(f"  ✓ Document listing working (returned {len(documents)} documents)")
        
        # Cleanup
        try:
            os.unlink(temp_file_path)
            print(f"\n🧹 Cleaned up temporary file")
        except:
            pass
            
        if success:
            print(f"\n🎉 SUCCESS: All source attribution tests passed!")
            return True
        else:
            print(f"\n❌ FAILED: Some source attribution tests failed!")
            return False
            
    except Exception as e:
        print(f"\n💥 Error during testing: {str(e)}")
        logger.error(f"Test error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_source_attribution()
    exit(0 if success else 1)