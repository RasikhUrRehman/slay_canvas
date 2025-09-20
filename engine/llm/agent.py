"""
AI Agent with OpenRouter LLM and Knowledge Base Search Tool
Provides an intelligent agent that can decide when to search the knowledge base
and generate queries automatically from user prompts.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

from engine.services.openrouter import OpenRouterClient
from engine.rag.rag_system import RAGSystem
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentResponse:
    """Response from the agent"""
    answer: str
    tools_used: List[str]
    reasoning: str
    confidence: float
    processing_time: float
    sources: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class KnowledgeBaseAgent:
    """
    AI Agent with OpenRouter LLM and knowledge base search capabilities.
    The agent can decide when to search the knowledge base and automatically
    generate appropriate search queries from user prompts.
    """
    
    def __init__(self, 
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 rag_collection_name: str = "agent_knowledge_base",
                 max_tokens: int = 1500,
                 temperature: float = 0.3):
        """
        Initialize the Knowledge Base Agent.
        
        Args:
            model: OpenRouter model to use (defaults to settings)
            api_key: OpenRouter API key (defaults to settings)
            rag_collection_name: Name for the RAG collection
            max_tokens: Maximum tokens for LLM responses
            temperature: Temperature for LLM generation
        """
        self.model = model or settings.OPENROUTER_MODEL
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize OpenRouter client
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
            
        self.llm_client = OpenRouterClient(
            model=self.model,
            api_key=self.api_key
        )
        logger.info(f"✓ OpenRouter client initialized with model: {self.model}")
        
        # Initialize RAG system for knowledge base
        try:
            self.rag_system = RAGSystem(
                collection_name=rag_collection_name,
                chunk_size=1000,
                chunk_overlap=200,
                top_k=5
            )
            logger.info("✓ RAG system initialized for knowledge base")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            self.rag_system = None
        
        # Register available tools
        self.tools = {
            "search_knowledge_base": self._search_knowledge_base_tool
        }
        
        # System prompt for the agent
        self.system_prompt = """You are an intelligent AI assistant with access to a knowledge base search tool.

Your capabilities:
1. You can search a knowledge base using the search_knowledge_base tool
2. You should decide when to use this tool based on the user's question
3. You can generate appropriate search queries automatically
4. You provide comprehensive answers combining your knowledge with search results

Tool Usage Guidelines:
- Use search_knowledge_base when the user asks about specific information that might be in documents
- Generate focused search queries that capture the key concepts from the user's question
- Always explain your reasoning for using or not using the tool
- If search results are found, incorporate them into your response and cite sources
- If no relevant results are found, rely on your general knowledge but mention the search attempt

Available Tools:
- search_knowledge_base(query: str) -> searches the knowledge base with the given query

When you decide to use a tool, format your response as:
REASONING: [Explain why you're using the tool]
TOOL_CALL: search_knowledge_base("your search query here")
RESPONSE: [Your final response incorporating any tool results]

If you don't need to use tools, just provide a direct response."""

    def _search_knowledge_base_tool(self, query: str) -> ToolResult:
        """
        Search the knowledge base with the given query.
        
        Args:
            query: Search query string
            
        Returns:
            ToolResult with search results
        """
        if not self.rag_system:
            return ToolResult(
                success=False,
                data=None,
                error="Knowledge base not available"
            )
        
        try:
            logger.info(f"Searching knowledge base with query: {query}")
            
            # Use RAG system to search (retrieval only)
            rag_response = self.rag_system.query(
                question=query,
                k=5,
                generate_answer=False  # Only retrieve, don't generate
            )
            
            if not rag_response.sources:
                return ToolResult(
                    success=True,
                    data=[],
                    metadata={"message": "No relevant documents found"}
                )
            
            # Format search results
            search_results = []
            for source in rag_response.sources:
                search_results.append({
                    "text": source.get("text", ""),
                    "similarity": source.get("similarity", 0.0),
                    "source": source.get("source", "Unknown"),
                    "metadata": source.get("metadata", {})
                })
            
            return ToolResult(
                success=True,
                data=search_results,
                metadata={
                    "query": query,
                    "total_results": len(search_results),
                    "processing_time": rag_response.processing_time
                }
            )
            
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

    def _should_use_knowledge_base(self, user_prompt: str) -> tuple[bool, str]:
        """
        Determine if the knowledge base should be searched and generate a query.
        
        Args:
            user_prompt: User's input prompt
            
        Returns:
            Tuple of (should_search, search_query)
        """
        # Use LLM to decide if knowledge base search is needed
        decision_prompt = f"""Analyze this user question and determine if it would benefit from searching a knowledge base of documents.

User Question: "{user_prompt}"

Consider:
1. Does this question ask for specific information that might be in documents?
2. Is this a factual question that could benefit from additional context?
3. Does this seem like a question about general knowledge vs. specific documented information?

Respond in this exact format:
DECISION: [YES or NO]
QUERY: [If YES, provide a focused search query; if NO, write "N/A"]
REASONING: [Brief explanation of your decision]"""

        try:
            response = self.llm_client.chat(
                messages=decision_prompt,
                max_tokens=200,
                system_prompt="You are a helpful assistant that decides when to search knowledge bases."
            )
            
            # Parse the response
            decision_match = re.search(r'DECISION:\s*(YES|NO)', response, re.IGNORECASE)
            query_match = re.search(r'QUERY:\s*(.+?)(?=\nREASONING:|$)', response, re.DOTALL)
            reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL)
            
            if decision_match:
                should_search = decision_match.group(1).upper() == "YES"
                search_query = query_match.group(1).strip() if query_match and should_search else ""
                reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
                
                logger.info(f"Decision: {'Search' if should_search else 'No search'}, Query: {search_query}")
                return should_search, search_query
            
        except Exception as e:
            logger.error(f"Failed to determine knowledge base usage: {e}")
        
        # Fallback: simple keyword-based decision
        knowledge_keywords = [
            "what is", "explain", "describe", "tell me about", "information about",
            "details", "documentation", "how to", "guide", "tutorial", "example"
        ]
        
        user_lower = user_prompt.lower()
        should_search = any(keyword in user_lower for keyword in knowledge_keywords)
        search_query = user_prompt if should_search else ""
        
        return should_search, search_query

    def process_query_stream(self, user_prompt: str, conversation_history: List[Dict[str, str]] = None):
        """
        Process a user query with streaming response, deciding whether to use tools.
        
        Args:
            user_prompt: User's input prompt
            conversation_history: Previous conversation messages for context
            
        Yields:
            String chunks of the response
        """
        tools_used = []
        sources = []
        reasoning = ""
        
        try:
            logger.info(f"Processing streaming user query: {user_prompt}")
            
            # Decide if knowledge base search is needed
            should_search, search_query = self._should_use_knowledge_base(user_prompt)
            
            knowledge_context = ""
            if should_search and search_query:
                reasoning = f"Searching knowledge base for: '{search_query}'"
                logger.info(reasoning)
                
                # Search knowledge base
                search_result = self._search_knowledge_base_tool(search_query)
                tools_used.append("search_knowledge_base")
                
                if search_result.success and search_result.data:
                    # Prepare context from search results
                    context_parts = []
                    for result in search_result.data:
                        context_parts.append(result["text"])
                        sources.append({
                            "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                            "similarity": result["similarity"],
                            "source": result["source"]
                        })
                    
                    knowledge_context = "\n\n".join(context_parts)
                    reasoning += f" Found {len(search_result.data)} relevant documents."
                else:
                    reasoning += " No relevant documents found in knowledge base."
            else:
                reasoning = "No knowledge base search needed for this query."
            
            # Build messages for streaming
            messages = []
            
            # Add system prompt
            system_prompt = self.system_prompt
            if knowledge_context:
                system_prompt += f"\n\nKnowledge Base Context:\n{knowledge_context}"
            
            messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history[-10:])  # Last 10 messages for context
            
            # Add current user message
            messages.append({"role": "user", "content": user_prompt})
            
            # Get streaming response from LLM
            for chunk in self.llm_client.chat_stream(
                messages=messages,
                max_tokens=self.max_tokens
            ):
                yield chunk
                
        except Exception as e:
            error_msg = f"Failed to process streaming query: {e}"
            logger.error(error_msg)
            yield f"Error: {error_msg}"

    def process_query(self, user_prompt: str) -> AgentResponse:
        """
        Process a user query, deciding whether to use tools and generating a response.
        
        Args:
            user_prompt: User's input prompt
            
        Returns:
            AgentResponse with the agent's response and metadata
        """
        start_time = datetime.now()
        tools_used = []
        sources = []
        reasoning = ""
        
        try:
            logger.info(f"Processing user query: {user_prompt}")
            
            # Decide if knowledge base search is needed
            should_search, search_query = self._should_use_knowledge_base(user_prompt)
            
            knowledge_context = ""
            if should_search and search_query:
                reasoning = f"Searching knowledge base for: '{search_query}'"
                logger.info(reasoning)
                
                # Search knowledge base
                search_result = self._search_knowledge_base_tool(search_query)
                tools_used.append("search_knowledge_base")
                
                if search_result.success and search_result.data:
                    # Prepare context from search results
                    context_parts = []
                    for result in search_result.data:
                        context_parts.append(result["text"])
                        sources.append({
                            "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                            "similarity": result["similarity"],
                            "source": result["source"]
                        })
                    
                    knowledge_context = "\n\n".join(context_parts)
                    reasoning += f" Found {len(search_result.data)} relevant documents."
                else:
                    reasoning += " No relevant documents found in knowledge base."
            else:
                reasoning = "No knowledge base search needed for this query."
            
            # Generate final response using LLM
            if knowledge_context:
                final_prompt = f"""Based on the following context from the knowledge base and the user's question, provide a comprehensive and accurate answer.

Context from Knowledge Base:
{knowledge_context}

User Question: {user_prompt}

Instructions:
- Use the context to provide accurate information
- If the context doesn't fully answer the question, supplement with your general knowledge
- Cite when you're using information from the knowledge base
- Be clear about what information comes from the knowledge base vs. your general knowledge

Answer:"""
            else:
                final_prompt = f"""User Question: {user_prompt}

Please provide a helpful and accurate response based on your knowledge."""

            # Get LLM response
            answer = self.llm_client.chat(
                messages=final_prompt,
                max_tokens=self.max_tokens,
                system_prompt=self.system_prompt
            )
            
            # Calculate confidence (simple heuristic)
            confidence = 0.8 if sources else 0.6
            if knowledge_context and len(sources) > 2:
                confidence = 0.9
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                answer=answer,
                tools_used=tools_used,
                reasoning=reasoning,
                confidence=confidence,
                processing_time=processing_time,
                sources=sources if sources else None
            )
            
        except Exception as e:
            error_msg = f"Failed to process query: {e}"
            logger.error(error_msg)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                answer=f"I apologize, but I encountered an error while processing your request: {error_msg}",
                tools_used=tools_used,
                reasoning=reasoning,
                confidence=0.0,
                processing_time=processing_time,
                error=error_msg
            )

    def add_document_to_knowledge_base(self, url: str) -> Dict[str, Any]:
        """
        Add a document to the knowledge base from URL.
        
        Args:
            url: URL of the document to add
            
        Returns:
            Dictionary with status information
        """
        if not self.rag_system:
            return {"success": False, "error": "Knowledge base not available"}
        
        try:
            status = self.rag_system.add_document_from_url(url)
            return {
                "success": status.status == "completed",
                "status": status.status,
                "chunks_created": status.chunks_created,
                "processing_time": status.processing_time,
                "error": status.error_message
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def add_text_to_knowledge_base(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add custom text to the knowledge base.
        
        Args:
            text: Text content to add
            metadata: Metadata for the text
            
        Returns:
            Dictionary with status information
        """
        if not self.rag_system:
            return {"success": False, "error": "Knowledge base not available"}
        
        try:
            status = self.rag_system.add_custom_text(text, metadata)
            return {
                "success": status.status == "completed",
                "status": status.status,
                "chunks_created": status.chunks_created,
                "processing_time": status.processing_time,
                "error": status.error_message
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with knowledge base statistics
        """
        if not self.rag_system:
            return {"error": "Knowledge base not available"}
        
        try:
            return self.rag_system.get_system_stats()
        except Exception as e:
            return {"error": str(e)}

    def list_knowledge_base_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the knowledge base.
        
        Returns:
            List of document information
        """
        if not self.rag_system:
            return []
        
        try:
            return self.rag_system.list_documents()
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []