"""
Knowledge Base Viewer
A utility to view the status and contents of any knowledge base by providing its name.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from tabulate import tabulate
from engine.rag.rag_system import RAGSystem

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce log noise
logger = logging.getLogger(__name__)

class KnowledgeBaseViewer:
    """Utility class to view knowledge base status and contents."""
    
    def __init__(self, knowledge_base_name: str):
        """
        Initialize the viewer with a knowledge base name.
        
        Args:
            knowledge_base_name: Name of the knowledge base collection to view
        """
        self.knowledge_base_name = knowledge_base_name
        self.rag_system = None
        
    def connect(self) -> bool:
        """
        Connect to the knowledge base.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.rag_system = RAGSystem(collection_name=self.knowledge_base_name)
            return True
        except Exception as e:
            print(f"❌ Error connecting to knowledge base '{self.knowledge_base_name}': {str(e)}")
            return False
    
    def get_basic_info(self) -> Dict[str, Any]:
        """Get basic information about the knowledge base."""
        if not self.rag_system:
            return {"error": "Not connected to knowledge base"}
        
        try:
            stats = self.rag_system.get_system_stats()
            documents = self.rag_system.list_documents()
            
            return {
                "knowledge_base_name": self.knowledge_base_name,
                "total_documents": len(documents),
                "total_chunks": stats.get("vector_store", {}).get("total_entities", 0),
                "vector_dimension": stats.get("vector_store", {}).get("dimension", 0),
                "collection_info": stats.get("vector_store", {}).get("collection_info", {}),
                "processing_status": stats.get("processing_status", {}),
                "system_components": stats.get("system_components", {})
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_documents_list(self) -> List[Dict[str, Any]]:
        """Get list of all documents in the knowledge base."""
        if not self.rag_system:
            return []
        
        try:
            return self.rag_system.list_documents()
        except Exception as e:
            print(f"❌ Error getting documents list: {str(e)}")
            return []
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents in the knowledge base.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results
        """
        if not self.rag_system:
            return []
        
        try:
            response = self.rag_system.query(query, k=k, generate_answer=False)
            return response.sources
        except Exception as e:
            print(f"❌ Error searching documents: {str(e)}")
            return []
    
    def display_basic_info(self):
        """Display basic information about the knowledge base."""
        print("=" * 80)
        print(f"KNOWLEDGE BASE STATUS: {self.knowledge_base_name}")
        print("=" * 80)
        
        info = self.get_basic_info()
        
        if "error" in info:
            print(f"❌ Error: {info['error']}")
            return
        
        print(f"📊 Knowledge Base Name: {info['knowledge_base_name']}")
        print(f"📄 Total Documents: {info['total_documents']}")
        print(f"🧩 Total Chunks: {info['total_chunks']}")
        print(f"📐 Vector Dimension: {info['vector_dimension']}")
        
        # Processing status
        processing_status = info.get('processing_status', {})
        if processing_status:
            print(f"\n📈 Processing Status:")
            for status, count in processing_status.items():
                print(f"  {status.capitalize()}: {count}")
        
        # System components
        components = info.get('system_components', {})
        if components:
            print(f"\n🔧 System Components:")
            for component, status in components.items():
                status_icon = "✅" if status == "active" else "❌"
                print(f"  {status_icon} {component.capitalize()}: {status}")
        
        # Collection info
        collection_info = info.get('collection_info', {})
        if collection_info:
            print(f"\n🗃️  Collection Details:")
            print(f"  Collection ID: {collection_info.get('collection_id', 'N/A')}")
            print(f"  Shards: {collection_info.get('num_shards', 'N/A')}")
            print(f"  Partitions: {collection_info.get('num_partitions', 'N/A')}")
    
    def display_documents_table(self):
        """Display documents in a formatted table."""
        documents = self.get_documents_list()
        
        if not documents:
            print("\n📭 No documents found in this knowledge base.")
            return
        
        print(f"\n📚 DOCUMENTS IN KNOWLEDGE BASE ({len(documents)} total)")
        print("-" * 80)
        
        # Prepare table data
        table_data = []
        for i, doc in enumerate(documents, 1):
            table_data.append([
                i,
                doc.get('title', 'N/A')[:40] + ('...' if len(doc.get('title', '')) > 40 else ''),
                doc.get('content_type', 'N/A'),
                doc.get('total_chunks', 0),
                f"{doc.get('total_characters', 0):,}",
                doc.get('extraction_time', 'N/A')[:19] if doc.get('extraction_time') else 'N/A'
            ])
        
        headers = ["#", "Title", "Type", "Chunks", "Characters", "Added"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all chunks with their metadata from the knowledge base."""
        try:
            if not self.rag_system or not self.rag_system.vector_store:
                return []
            
            # Get all documents (chunks) from vector store
            documents = self.rag_system.vector_store.list_all_documents()
            
            chunks = []
            for text, metadata in documents:
                chunk_data = {
                    'text': text,
                    'source_url': metadata.get('source_url', ''),
                    'title': metadata.get('title', ''),
                    'content_type': metadata.get('content_type', ''),
                    'chunk_index': metadata.get('chunk_index', 0),
                    'total_chunks': metadata.get('total_chunks', 0),
                    'chunk_size': metadata.get('chunk_size', 0),
                    'extraction_time': metadata.get('extraction_time', ''),
                    'transcription_type': metadata.get('transcription_type', '')
                }
                chunks.append(chunk_data)
            
            # Sort by source_url and chunk_index for better organization
            chunks.sort(key=lambda x: (x['source_url'], x['chunk_index']))
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting chunks: {e}")
            return []
    
    def display_chunks_table(self):
        """Display all chunks with their metadata in a formatted table."""
        chunks = self.get_all_chunks()
        
        if not chunks:
            print("\n📭 No chunks found in this knowledge base.")
            return
        
        print(f"\n🧩 CHUNKS IN KNOWLEDGE BASE ({len(chunks)} total)")
        print("-" * 120)
        
        # Prepare table data
        table_data = []
        for i, chunk in enumerate(chunks, 1):
            # Truncate text for display
            text_preview = chunk['text'][:60].replace('\n', ' ').replace('\r', ' ')
            if len(chunk['text']) > 60:
                text_preview += '...'
            
            table_data.append([
                i,
                chunk.get('title', 'N/A')[:25] + ('...' if len(chunk.get('title', '')) > 25 else ''),
                f"{chunk.get('chunk_index', 0) + 1}/{chunk.get('total_chunks', 0)}",
                chunk.get('content_type', 'N/A'),
                len(chunk['text']),
                text_preview,
                chunk.get('extraction_time', 'N/A')[:10] if chunk.get('extraction_time') else 'N/A'
            ])
        
        headers = ["#", "Title", "Chunk", "Type", "Size", "Preview", "Added"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Show detailed metadata for first few chunks as example
        print(f"\n🔍 DETAILED METADATA (showing first 3 chunks):")
        print("-" * 80)
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\nChunk #{i}:")
            print(f"  📄 Title: {chunk.get('title', 'N/A')}")
            print(f"  🔗 Source: {chunk.get('source_url', 'N/A')}")
            print(f"  📝 Type: {chunk.get('content_type', 'N/A')}")
            print(f"  🧩 Chunk: {chunk.get('chunk_index', 0) + 1} of {chunk.get('total_chunks', 0)}")
            print(f"  📏 Size: {len(chunk['text'])} characters")
            if chunk.get('transcription_type'):
                print(f"  🎤 Transcription: {chunk.get('transcription_type', 'N/A')}")
            print(f"  📅 Added: {chunk.get('extraction_time', 'N/A')}")
            print(f"  📖 Content: {chunk['text'][:200]}{'...' if len(chunk['text']) > 200 else ''}")
            print("-" * 40)
    
    def display_search_results(self, query: str, k: int = 5):
        """Display search results for a query."""
        print(f"\n🔍 SEARCH RESULTS FOR: '{query}'")
        print("-" * 80)
        
        results = self.search_documents(query, k)
        
        if not results:
            print("❌ No results found.")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\n[Result {i}]")
            print(f"📄 Title: {result.get('metadata', {}).get('title', 'N/A')}")
            print(f"📂 Category: {result.get('metadata', {}).get('category', 'N/A')}")
            print(f"🎯 Relevance Score: {result.get('score', 0):.4f}")
            print(f"📝 Content Preview: {result.get('text', '')[:200]}...")
            print("-" * 40)
    
    def interactive_mode(self):
        """Run in interactive mode for exploring the knowledge base."""
        if not self.connect():
            return
        
        self.display_basic_info()
        self.display_documents_table()
        
        print(f"\n🔍 INTERACTIVE SEARCH MODE")
        print("Enter search queries to explore the knowledge base. Type 'quit' to exit.")
        print("-" * 80)
        
        while True:
            try:
                query = input("\n🔍 Search query (or 'quit' to exit): ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                self.display_search_results(query)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

def main():
    """Main function to run the knowledge base viewer."""
    print("🔍 KNOWLEDGE BASE VIEWER")
    print("=" * 80)
    
    # Get knowledge base name from command line or user input
    if len(sys.argv) > 1:
        kb_name = sys.argv[1]
    else:
        kb_name = input("Enter knowledge base name: ").strip()
    
    if not kb_name:
        print("❌ No knowledge base name provided.")
        return
    
    # Create viewer and run
    viewer = KnowledgeBaseViewer(kb_name)
    
    # Check if user wants interactive mode
    if len(sys.argv) > 2 and sys.argv[2] == "--interactive":
        viewer.interactive_mode()
    else:
        # Show basic info, documents, and chunks
        if viewer.connect():
            viewer.display_basic_info()
            viewer.display_documents_table()
            viewer.display_chunks_table()
            
            # Ask if user wants to search
            try:
                search_query = input(f"\n🔍 Enter a search query (or press Enter to skip): ").strip()
                if search_query:
                    viewer.display_search_results(search_query)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")

def quick_status(kb_name: str) -> Dict[str, Any]:
    """
    Quick function to get knowledge base status without interactive mode.
    
    Args:
        kb_name: Knowledge base name
        
    Returns:
        Dictionary with status information
    """
    viewer = KnowledgeBaseViewer(kb_name)
    if viewer.connect():
        return viewer.get_basic_info()
    else:
        return {"error": f"Could not connect to knowledge base '{kb_name}'"}

if __name__ == "__main__":
    main()