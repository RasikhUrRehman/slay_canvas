"""
Test file to create a new knowledge base and add documents to it.
This script demonstrates how to set up a knowledge base with sample documents.
"""

import logging
import json
from datetime import datetime
from engine.rag.rag_system import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Knowledge base configuration
KNOWLEDGE_BASE_NAME = "test_knowledge_base_2024"
KNOWLEDGE_BASE_CONFIG_FILE = "knowledge_base_config.json"

def create_knowledge_base_with_documents():
    """
    Create a new knowledge base and add sample documents to it.
    """
    print(f"Creating knowledge base: {KNOWLEDGE_BASE_NAME}")
    
    # Initialize RAG system with our custom collection name
    rag_system = RAGSystem(
        collection_name=KNOWLEDGE_BASE_NAME,
        chunk_size=800,
        chunk_overlap=100,
        top_k=5
    )
    
    # Sample documents to add to the knowledge base
    sample_documents = [
        {
            "text": """
            Artificial Intelligence (AI) is a branch of computer science that aims to create 
            intelligent machines that can perform tasks that typically require human intelligence. 
            These tasks include learning, reasoning, problem-solving, perception, and language 
            understanding. AI systems can be categorized into narrow AI, which is designed for 
            specific tasks, and general AI, which would have human-like cognitive abilities 
            across various domains.
            """,
            "metadata": {
                "title": "Introduction to Artificial Intelligence",
                "category": "AI Basics",
                "source": "Educational Content",
                "date_added": datetime.now().isoformat()
            }
        },
        {
            "text": """
            Machine Learning is a subset of artificial intelligence that enables computers to 
            learn and improve from experience without being explicitly programmed. It involves 
            algorithms that can identify patterns in data and make predictions or decisions 
            based on that data. Common types of machine learning include supervised learning, 
            unsupervised learning, and reinforcement learning. Applications include image 
            recognition, natural language processing, and recommendation systems.
            """,
            "metadata": {
                "title": "Machine Learning Fundamentals",
                "category": "Machine Learning",
                "source": "Educational Content",
                "date_added": datetime.now().isoformat()
            }
        },
        {
            "text": """
            Natural Language Processing (NLP) is a field of AI that focuses on the interaction 
            between computers and human language. It involves developing algorithms and models 
            that can understand, interpret, and generate human language in a valuable way. 
            NLP applications include chatbots, language translation, sentiment analysis, 
            text summarization, and speech recognition. Modern NLP heavily relies on deep 
            learning techniques and transformer architectures.
            """,
            "metadata": {
                "title": "Natural Language Processing Overview",
                "category": "NLP",
                "source": "Educational Content",
                "date_added": datetime.now().isoformat()
            }
        },
        {
            "text": """
            Deep Learning is a subset of machine learning that uses artificial neural networks 
            with multiple layers to model and understand complex patterns in data. These deep 
            neural networks are inspired by the structure and function of the human brain. 
            Deep learning has revolutionized many fields including computer vision, natural 
            language processing, and speech recognition. Popular architectures include 
            convolutional neural networks (CNNs), recurrent neural networks (RNNs), and 
            transformers.
            """,
            "metadata": {
                "title": "Deep Learning Concepts",
                "category": "Deep Learning",
                "source": "Educational Content",
                "date_added": datetime.now().isoformat()
            }
        },
        {
            "text": """
            Computer Vision is a field of AI that enables computers to interpret and understand 
            visual information from the world. It involves developing algorithms that can 
            process, analyze, and understand digital images and videos. Applications include 
            object detection, facial recognition, medical image analysis, autonomous vehicles, 
            and augmented reality. Modern computer vision systems often use convolutional 
            neural networks and other deep learning techniques to achieve high accuracy.
            """,
            "metadata": {
                "title": "Computer Vision Applications",
                "category": "Computer Vision",
                "source": "Educational Content",
                "date_added": datetime.now().isoformat()
            }
        }
    ]
    
    # Add documents to the knowledge base
    print("Adding documents to the knowledge base...")
    document_statuses = []
    
    for i, doc in enumerate(sample_documents, 1):
        print(f"Adding document {i}/{len(sample_documents)}: {doc['metadata']['title']}")
        
        try:
            status = rag_system.add_custom_text(
                text=doc["text"].strip(),
                metadata=doc["metadata"]
            )
            document_statuses.append(status)
            print(f"  ✓ Added successfully - {status.chunks_created} chunks created")
            
        except Exception as e:
            print(f"  ✗ Error adding document: {str(e)}")
            logger.error(f"Error adding document {i}: {str(e)}")
    
    # Save knowledge base configuration
    config = {
        "knowledge_base_name": KNOWLEDGE_BASE_NAME,
        "created_at": datetime.now().isoformat(),
        "total_documents": len(sample_documents),
        "successful_documents": len([s for s in document_statuses if s.status == "completed"]),
        "collection_name": KNOWLEDGE_BASE_NAME,
        "chunk_size": 800,
        "chunk_overlap": 100,
        "top_k": 5
    }
    
    with open(KNOWLEDGE_BASE_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nKnowledge base configuration saved to: {KNOWLEDGE_BASE_CONFIG_FILE}")
    
    # Display system stats
    try:
        stats = rag_system.get_system_stats()
        print(f"\nKnowledge Base Stats:")
        print(f"  Collection Name: {KNOWLEDGE_BASE_NAME}")
        print(f"  Total Documents: {stats.get('total_documents', 'N/A')}")
        print(f"  Total Chunks: {stats.get('total_chunks', 'N/A')}")
        print(f"  Vector Store Stats: {stats.get('vector_store', {})}")
        
    except Exception as e:
        print(f"Error getting stats: {str(e)}")
    
    return KNOWLEDGE_BASE_NAME, config

def main():
    """Main function to run the knowledge base setup."""
    print("=" * 60)
    print("KNOWLEDGE BASE SETUP TEST")
    print("=" * 60)
    
    try:
        kb_name, config = create_knowledge_base_with_documents()
        
        print(f"\n✓ Knowledge base '{kb_name}' created successfully!")
        print(f"✓ Configuration saved to '{KNOWLEDGE_BASE_CONFIG_FILE}'")
        print(f"✓ Ready to use with AI agents")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error creating knowledge base: {str(e)}")
        logger.error(f"Knowledge base creation failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "=" * 60)
        print("SETUP COMPLETED SUCCESSFULLY")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SETUP FAILED")
        print("=" * 60)