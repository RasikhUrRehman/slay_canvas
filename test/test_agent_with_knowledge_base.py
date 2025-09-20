"""
Test file to initialize an AI agent with the created knowledge base and test question answering.
This script demonstrates how to use the KnowledgeBaseAgent with a custom knowledge base.
"""

import logging
import json
import os
from datetime import datetime
from engine.llm.agent import KnowledgeBaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration file created by the setup script
KNOWLEDGE_BASE_CONFIG_FILE = "knowledge_base_config.json"

def load_knowledge_base_config():
    """Load the knowledge base configuration from the setup script."""
    if not os.path.exists(KNOWLEDGE_BASE_CONFIG_FILE):
        raise FileNotFoundError(
            f"Knowledge base configuration file '{KNOWLEDGE_BASE_CONFIG_FILE}' not found. "
            "Please run 'test_knowledge_base_setup.py' first."
        )
    
    with open(KNOWLEDGE_BASE_CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    return config

def initialize_agent_with_knowledge_base():
    """Initialize the AI agent with the created knowledge base."""
    
    # Load knowledge base configuration
    config = load_knowledge_base_config()
    kb_name = config["knowledge_base_name"]
    
    print(f"Initializing agent with knowledge base: {kb_name}")
    
    # Initialize the KnowledgeBaseAgent with our custom knowledge base
    agent = KnowledgeBaseAgent(
        rag_collection_name=kb_name,
        max_tokens=2000,
        temperature=0.3
    )
    
    print(f"✓ Agent initialized successfully")
    
    # Get knowledge base stats to verify it's working
    try:
        stats = agent.get_knowledge_base_stats()
        print(f"✓ Knowledge base stats: {stats}")
    except Exception as e:
        print(f"Warning: Could not get knowledge base stats: {str(e)}")
    
    return agent, config

def test_agent_questions(agent):
    """Test the agent with various questions about the knowledge base content."""
    
    # Test questions covering different topics in the knowledge base
    test_questions = [
        "What is artificial intelligence?",
        "Explain the difference between machine learning and deep learning.",
        "What are the main applications of natural language processing?",
        "How does computer vision work?",
        "What types of machine learning are there?",
        "What are neural networks used for?",
        "Can you compare supervised and unsupervised learning?",
        "What is the role of transformers in NLP?",
        "How is AI different from traditional programming?",
        "What are some real-world applications of deep learning?"
    ]
    
    print("\n" + "=" * 80)
    print("TESTING AGENT WITH KNOWLEDGE BASE QUESTIONS")
    print("=" * 80)
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[Question {i}/{len(test_questions)}]")
        print(f"Q: {question}")
        print("-" * 60)
        
        try:
            # Process the query using the agent
            response = agent.process_query(question)
            
            print(f"A: {response.answer}")
            print(f"\nTools used: {', '.join(response.tools_used) if response.tools_used else 'None'}")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Processing time: {response.processing_time:.2f}s")
            
            if response.sources:
                print(f"Sources found: {len(response.sources)}")
                for j, source in enumerate(response.sources[:2], 1):  # Show first 2 sources
                    title = source.get('metadata', {}).get('title', 'Unknown')
                    category = source.get('metadata', {}).get('category', 'Unknown')
                    print(f"  {j}. {title} ({category})")
            
            results.append({
                "question": question,
                "answer": response.answer,
                "tools_used": response.tools_used,
                "confidence": response.confidence,
                "processing_time": response.processing_time,
                "sources_count": len(response.sources) if response.sources else 0,
                "success": True
            })
            
        except Exception as e:
            print(f"✗ Error processing question: {str(e)}")
            logger.error(f"Error processing question '{question}': {str(e)}")
            
            results.append({
                "question": question,
                "error": str(e),
                "success": False
            })
    
    return results

def save_test_results(results, config):
    """Save the test results to a file."""
    
    test_results = {
        "test_timestamp": datetime.now().isoformat(),
        "knowledge_base_name": config["knowledge_base_name"],
        "knowledge_base_created": config["created_at"],
        "total_questions": len(results),
        "successful_questions": len([r for r in results if r.get("success", False)]),
        "failed_questions": len([r for r in results if not r.get("success", False)]),
        "results": results
    }
    
    results_file = "agent_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Test results saved to: {results_file}")
    return results_file

def display_summary(results):
    """Display a summary of the test results."""
    
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total questions: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        avg_confidence = sum(r["confidence"] for r in successful) / len(successful)
        avg_processing_time = sum(r["processing_time"] for r in successful) / len(successful)
        total_sources = sum(r["sources_count"] for r in successful)
        
        print(f"Average confidence: {avg_confidence:.2f}")
        print(f"Average processing time: {avg_processing_time:.2f}s")
        print(f"Total sources retrieved: {total_sources}")
        
        # Count tool usage
        tools_used = {}
        for r in successful:
            for tool in r.get("tools_used", []):
                tools_used[tool] = tools_used.get(tool, 0) + 1
        
        if tools_used:
            print(f"Tool usage:")
            for tool, count in tools_used.items():
                print(f"  {tool}: {count} times")
    
    if failed:
        print(f"\nFailed questions:")
        for r in failed:
            print(f"  - {r['question']}: {r.get('error', 'Unknown error')}")

def main():
    """Main function to run the agent test."""
    print("=" * 80)
    print("AI AGENT WITH KNOWLEDGE BASE TEST")
    print("=" * 80)
    
    try:
        # Initialize agent with knowledge base
        agent, config = initialize_agent_with_knowledge_base()
        
        # Test the agent with questions
        results = test_agent_questions(agent)
        
        # Save and display results
        results_file = save_test_results(results, config)
        display_summary(results)
        
        print(f"\n✓ Agent testing completed successfully!")
        print(f"✓ Results saved to '{results_file}'")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during agent testing: {str(e)}")
        logger.error(f"Agent testing failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "=" * 80)
        print("AGENT TESTING COMPLETED SUCCESSFULLY")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("AGENT TESTING FAILED")
        print("=" * 80)