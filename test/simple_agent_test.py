"""
Simple test to verify the agent can use the knowledge base.
This is a minimal test to check basic functionality.
"""

import json
from engine.llm.agent import KnowledgeBaseAgent

def simple_test():
    """Simple test of the agent with the knowledge base."""
    
    # Load the knowledge base configuration
    with open("knowledge_base_config.json", 'r') as f:
        config = json.load(f)
    
    kb_name = config["knowledge_base_name"]
    print(f"Testing agent with knowledge base: {kb_name}")
    
    # Initialize the agent
    agent = KnowledgeBaseAgent(
        rag_collection_name=kb_name,
        max_tokens=1000,
        temperature=0.3
    )
    
    # Test with a simple question
    question = "What is artificial intelligence?"
    print(f"\nQuestion: {question}")
    
    try:
        response = agent.process_query(question)
        print(f"\nAnswer: {response.answer}")
        print(f"Tools used: {response.tools_used}")
        print(f"Confidence: {response.confidence}")
        
        if response.sources:
            print(f"Sources found: {len(response.sources)}")
        
        print("\n✓ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("SIMPLE AGENT TEST")
    print("=" * 50)
    
    success = simple_test()
    
    if success:
        print("\n✓ Agent is working with the knowledge base!")
    else:
        print("\n✗ Agent test failed.")