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
                 vector_store=None,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 rag_collection_name: str = "agent_knowledge_base",
                 max_tokens: int = 1500,
                 temperature: float = 0.3):
        """
        Initialize the Knowledge Base Agent.
        
        Args:
            vector_store: Existing VectorStore instance to use (optional)
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
        if vector_store:
            # Use provided vector store directly
            self.vector_store = vector_store
            self.rag_system = None
            logger.info("✓ Using provided VectorStore instance")
        else:
            # Initialize RAG system with collection name
            try:
                self.rag_system = RAGSystem(
                    collection_name=rag_collection_name,
                    chunk_size=1000,
                    chunk_overlap=200,
                    top_k=5
                )
                self.vector_store = self.rag_system.vector_store
                logger.info("✓ RAG system initialized for knowledge base")
            except Exception as e:
                logger.error(f"Failed to initialize RAG system: {e}")
                self.rag_system = None
                self.vector_store = None
        
        # Register available tools
        self.tools = {
            "search_knowledge_base": self._search_knowledge_base_tool,
            "summarize_knowledge_base": self._summarize_knowledge_base_tool,
            "generate_idea": self._generate_idea_tool
        }
        
        # System prompt for the agent
        self.system_prompt = """You are an intelligent AI assistant with access to knowledge base tools.

        Your capabilities:
        1. You can search a knowledge base using the search_knowledge_base tool
        2. You can summarize all content in the knowledge base using the summarize_knowledge_base tool
        3. You can generate creative ideas based on all documents using the generate_idea tool
        4. You should decide when to use these tools based on the user's question
        5. You can generate appropriate search queries automatically
        6. You provide comprehensive answers combining your knowledge with tool results

        Tool Usage Guidelines:
        - Use summarize_knowledge_base when the user asks for an overview, summary, or wants to know what's in the knowledge base
        - Use search_knowledge_base when the user asks about specific information that might be in documents
        - Use generate_idea when the user wants creative content, new ideas, or synthesis based on all available documents
        - Generate focused search queries that capture the key concepts from the user's question
        - Always explain your reasoning for using or not using tools
        - If tool results are found, incorporate them into your response and cite sources
        - If no relevant results are found, rely on your general knowledge but mention the tool attempt

        Available Tools:
        - search_knowledge_base(query: str) -> searches the knowledge base with the given query
        - summarize_knowledge_base() -> provides an overview of all content in the knowledge base
        - generate_idea(topic: str, content_type: str) -> generates creative ideas based on all documents

        When you decide to use a tool, format your response as:
        REASONING: [Explain why you're using the tool]
        TOOL_CALL: [tool_name]("parameters if needed")
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

    def _generate_idea_tool(self, topic: str = "", content_type: str = "article") -> ToolResult:
        """
        Generate creative ideas based on all documents in the knowledge base.
        
        Args:
            topic: Optional topic to focus on
            content_type: Type of content to generate (article, blog_post, summary, etc.)
            
        Returns:
            ToolResult with generated idea content
        """
        try:
            logger.info(f"Generating idea for topic: '{topic}', content_type: '{content_type}'")
            
            # Get all documents from the knowledge base
            if self.rag_system and self.rag_system.vector_store:
                all_documents = self.rag_system.vector_store.list_all_documents()
            elif self.vector_store:
                all_documents = self.vector_store.list_all_documents()
            else:
                return ToolResult(
                    success=False,
                    error="No vector store available for idea generation",
                    data=None
                )
            
            if not all_documents:
                return ToolResult(
                    success=False,
                    error="No documents found in knowledge base for idea generation",
                    data=None
                )
            
            # Prepare document summaries for idea generation
            document_summaries = []
            unique_sources = set()
            
            for text, metadata in all_documents[:20]:  # Limit to 20 documents
                source_url = metadata.get('source_url', '')
                title = metadata.get('title', 'Untitled')
                doc_content_type = metadata.get('content_type', 'unknown')
                
                # Avoid duplicate sources
                if source_url not in unique_sources:
                    unique_sources.add(source_url)
                    # Take first 300 characters as summary
                    summary = text[:300] + "..." if len(text) > 300 else text
                    document_summaries.append({
                        'title': title,
                        'content_type': doc_content_type,
                        'summary': summary,
                        'source_url': source_url
                    })
            
            # Create the idea generation prompt
            topic_context = f" focusing on the topic: {topic}" if topic else ""
            content_type_instruction = f"Format the output as a {content_type}."
            
            idea_prompt = f"""Based on the following documents from the knowledge base, generate new creative content ideas{topic_context}.

Available Documents:
"""
            
            for i, doc in enumerate(document_summaries, 1):
                idea_prompt += f"\n{i}. Title: {doc['title']}\n   Type: {doc['content_type']}\n   Summary: {doc['summary']}\n"
            
            idea_prompt += f"""

Task: Generate innovative and creative content based on the themes, concepts, and information from these documents. {content_type_instruction}

Requirements:
- Create original content that synthesizes information from multiple sources
- Identify patterns, connections, and insights across the documents
- Suggest new perspectives or applications of the existing knowledge
- Keep the response under 800 words
- Be creative and think outside the box while staying grounded in the source material

Generate your response now:"""
            
            # Generate the idea using the LLM
            response = self.llm_client.chat(
                messages=idea_prompt,
                max_tokens=1000,
                system_prompt="You are a creative content generator that synthesizes information from multiple sources."
            )
            
            if response and response.strip():
                return ToolResult(
                    success=True,
                    data={
                        "generated_idea": response.strip(),
                        "topic": topic,
                        "content_type": content_type,
                        "sources_used": len(document_summaries),
                        "total_documents": len(all_documents)
                    },
                    metadata={
                        "tool": "generate_idea",
                        "sources": [doc['source_url'] for doc in document_summaries if doc['source_url']]
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    error="Failed to generate idea - empty response from LLM",
                    data=None
                )
                
        except Exception as e:
            logger.error(f"Error in generate_idea_tool: {e}")
            return ToolResult(
                success=False,
                error=f"Failed to generate idea: {str(e)}",
                data=None
            )

    def _summarize_knowledge_base_tool(self, max_chunks: int = 50) -> ToolResult:
        """
        Summarize all documents in the knowledge base by fetching chunks from all documents.
        This gives the agent an overview of what the knowledge base contains.
        
        Args:
            max_chunks: Maximum number of chunks to include in summary (default: 50)
            
        Returns:
            ToolResult with knowledge base summary
        """
        if not self.rag_system:
            return ToolResult(
                success=False,
                data=None,
                error="Knowledge base not available"
            )
        
        try:
            logger.info(f"Summarizing knowledge base with max {max_chunks} chunks")
            
            # Get all documents from the knowledge base
            if self.rag_system and self.rag_system.vector_store:
                all_documents = self.rag_system.vector_store.list_all_documents()
            elif self.vector_store:
                all_documents = self.vector_store.list_all_documents()
            else:
                return ToolResult(
                    success=False,
                    error="No vector store available for summarization",
                    data=None
                )
            
            if not all_documents:
                return ToolResult(
                    success=True,
                    data={
                        "summary": "The knowledge base is empty - no documents have been added yet.",
                        "total_chunks": 0,
                        "documents": [],
                        "content_types": [],
                        "sources": []
                    },
                    metadata={"message": "Knowledge base is empty"}
                )
            
            # Limit the number of chunks if specified
            if max_chunks and len(all_documents) > max_chunks:
                # Take a representative sample from different documents
                documents_by_source = {}
                for text, metadata in all_documents:
                    source = metadata.get("source_url", "Unknown")
                    if source not in documents_by_source:
                        documents_by_source[source] = []
                    documents_by_source[source].append((text, metadata))
                
                # Take chunks evenly from different sources
                selected_documents = []
                chunks_per_source = max(1, max_chunks // len(documents_by_source))
                remaining_chunks = max_chunks
                
                for source, chunks in documents_by_source.items():
                    take_count = min(chunks_per_source, len(chunks), remaining_chunks)
                    selected_documents.extend(chunks[:take_count])
                    remaining_chunks -= take_count
                    if remaining_chunks <= 0:
                        break
                
                all_documents = selected_documents
            
            # Analyze the documents
            sources = set()
            content_types = set()
            document_summaries = {}
            
            for text, metadata in all_documents:
                source = metadata.get("source_url", "Unknown")
                content_type = metadata.get("content_type", "Unknown")
                original_filename = metadata.get("original_filename", source)
                
                sources.add(original_filename or source)
                content_types.add(content_type)
                
                # Group chunks by source for document-level summaries
                if source not in document_summaries:
                    document_summaries[source] = {
                        "title": metadata.get("title", original_filename or source),
                        "content_type": content_type,
                        "chunks": [],
                        "total_length": 0
                    }
                
                document_summaries[source]["chunks"].append(text)
                document_summaries[source]["total_length"] += len(text)
            
            # Create document summaries
            documents_info = []
            for source, info in document_summaries.items():
                chunk_count = len(info["chunks"])
                avg_chunk_length = info["total_length"] // chunk_count if chunk_count > 0 else 0
                
                # Get a sample of content from the first few chunks
                sample_content = ""
                for i, chunk in enumerate(info["chunks"][:3]):  # First 3 chunks
                    sample_content += f"Chunk {i+1}: {chunk[:200]}{'...' if len(chunk) > 200 else ''}\n\n"
                
                documents_info.append({
                    "source": info["title"],
                    "content_type": info["content_type"],
                    "chunk_count": chunk_count,
                    "avg_chunk_length": avg_chunk_length,
                    "sample_content": sample_content.strip()
                })
            
            # Generate overall summary using LLM if available
            summary_text = ""
            if self.llm_client:
                try:
                    # Prepare content for summarization
                    content_for_summary = ""
                    for doc_info in documents_info:
                        content_for_summary += f"Document: {doc_info['source']} ({doc_info['content_type']})\n"
                        content_for_summary += f"Chunks: {doc_info['chunk_count']}\n"
                        content_for_summary += f"Sample content:\n{doc_info['sample_content']}\n\n"
                    
                    summary_prompt = f"""Analyze the following knowledge base contents and provide a comprehensive summary of what information is available:

{content_for_summary}

Please provide:
1. An overview of the types of documents and content available
2. Key topics and themes covered
3. The scope and breadth of information
4. Any notable patterns or categories of content

Summary:"""

                    summary_text = self.llm_client.chat(
                        messages=summary_prompt,
                        max_tokens=800,
                        system_prompt="You are a helpful assistant that summarizes knowledge base contents."
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to generate LLM summary: {e}")
                    summary_text = "Unable to generate detailed summary, but document information is available below."
            else:
                summary_text = f"Knowledge base contains {len(all_documents)} chunks from {len(sources)} sources covering {len(content_types)} content types."
            
            return ToolResult(
                success=True,
                data={
                    "summary": summary_text,
                    "total_chunks": len(all_documents),
                    "total_sources": len(sources),
                    "documents": documents_info,
                    "content_types": list(content_types),
                    "sources": list(sources)
                },
                metadata={
                    "chunks_analyzed": len(all_documents),
                    "max_chunks_requested": max_chunks,
                    "sources_found": len(sources)
                }
            )
            
        except Exception as e:
            logger.error(f"Knowledge base summarization failed: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

    def _should_use_knowledge_base(self, user_prompt: str) -> tuple[bool, str, str]:
        """
        Determine if the knowledge base should be used and which tool to use.
        
        Args:
            user_prompt: User's input prompt
            
        Returns:
            Tuple of (should_use_kb, tool_name, search_query)
        """
        # Use LLM to decide if knowledge base should be used and which tool
        decision_prompt = f"""Analyze this user question and determine if it would benefit from using the knowledge base and which tool to use.

User Question: "{user_prompt}"

Available tools:
1. search_knowledge_base - Search for specific information in documents
2. summarize_knowledge_base - Get an overview of all content in the knowledge base
3. generate_idea - Generate creative ideas based on all documents in the knowledge base

Consider:
1. Does this question ask for a summary or overview of all content? (use summarize_knowledge_base)
2. Does this question ask for specific information that might be in documents? (use search_knowledge_base)
3. Does this question ask for creative ideas, content generation, or synthesis based on available content? (use generate_idea)
4. Is this a general question that doesn't need the knowledge base? (use neither)

Keywords that indicate summarization: "summarize", "summary", "overview", "what do you have", "what's in your knowledge base", "content available", "all documents"

Keywords that indicate idea generation: "create idea", "generate idea", "new idea", "creative content", "based on content", "synthesize", "combine information", "new perspective", "innovative", "brainstorm", "inspiration"

Respond in this exact format:
DECISION: [YES or NO]
TOOL: [search_knowledge_base, summarize_knowledge_base, generate_idea, or N/A]
QUERY: [If using search tool, provide a focused search query; if using generate_idea, write the topic or "general"; otherwise write "N/A"]
REASONING: [Brief explanation of your decision]"""

        try:
            response = self.llm_client.chat(
                messages=decision_prompt,
                max_tokens=200,
                system_prompt="You are a helpful assistant that decides when to search knowledge bases."
            )
            
            # Parse the response
            decision_match = re.search(r'DECISION:\s*(YES|NO)', response, re.IGNORECASE)
            tool_match = re.search(r'TOOL:\s*(.+?)(?=\nQUERY:|$)', response, re.DOTALL)
            query_match = re.search(r'QUERY:\s*(.+?)(?=\nREASONING:|$)', response, re.DOTALL)
            reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL)
            
            if decision_match:
                should_use_kb = decision_match.group(1).upper() == "YES"
                tool_name = tool_match.group(1).strip() if tool_match and should_use_kb else ""
                search_query = query_match.group(1).strip() if query_match and should_use_kb else ""
                reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
                
                logger.info(f"Decision: {'Use KB' if should_use_kb else 'No KB'}, Tool: {tool_name}, Query: {search_query}")
                return should_use_kb, tool_name, search_query
            
        except Exception as e:
            logger.error(f"Failed to determine knowledge base usage: {e}")
        
        # Fallback: simple keyword-based decision
        summarize_keywords = [
            "summarize", "summary", "summarise", "overview", "what do you have", 
            "what's in your knowledge base", "content available", "all documents",
            "what content", "what information", "knowledge base content"
        ]
        
        search_keywords = [
            "what is", "explain", "describe", "tell me about", "information about",
            "details", "documentation", "how to", "guide", "tutorial", "example"
        ]
        
        user_lower = user_prompt.lower()
        
        # Check for summarization keywords first
        if any(keyword in user_lower for keyword in summarize_keywords):
            return True, "summarize_knowledge_base", ""
        
        # Check for search keywords
        if any(keyword in user_lower for keyword in search_keywords):
            return True, "search_knowledge_base", user_prompt
        
        return False, "", ""

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
            
            # Decide if knowledge base should be used and which tool
            should_use_kb, tool_name, search_query = self._should_use_knowledge_base(user_prompt)
            
            knowledge_context = ""
            if should_use_kb and tool_name:
                if tool_name == "search_knowledge_base" and search_query:
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
                        
                elif tool_name == "summarize_knowledge_base":
                    reasoning = "Summarizing all content in the knowledge base"
                    logger.info(reasoning)
                    
                    # Summarize knowledge base
                    summary_result = self._summarize_knowledge_base_tool()
                    tools_used.append("summarize_knowledge_base")
                    
                    if summary_result.success and summary_result.data:
                        # Use the summary as context
                        summary_data = summary_result.data
                        knowledge_context = f"Knowledge Base Summary:\n{summary_data['summary']}\n\n"
                        knowledge_context += f"Total Sources: {summary_data['total_sources']}\n"
                        knowledge_context += f"Content Types: {', '.join(summary_data['content_types'])}\n\n"
                        
                        # Add document details
                        knowledge_context += "Document Details:\n"
                        for doc in summary_data['documents']:
                            knowledge_context += f"- {doc['source']} ({doc['content_type']}): {doc['chunk_count']} chunks\n"
                        
                        reasoning += f" Analyzed {summary_data['total_chunks']} chunks from {summary_data['total_sources']} sources."
                    else:
                        reasoning += " Failed to summarize knowledge base."
                        
                elif tool_name == "generate_idea":
                    reasoning = f"Generating creative ideas based on all knowledge base content"
                    if search_query and search_query.lower() != "general":
                        reasoning += f" with focus on: '{search_query}'"
                    logger.info(reasoning)
                    
                    # Generate idea based on all documents
                    topic = search_query if search_query and search_query.lower() != "general" else ""
                    idea_result = self._generate_idea_tool(topic=topic, content_type="article")
                    tools_used.append("generate_idea")
                    
                    if idea_result.success and idea_result.data:
                        # Use the generated idea as the main response
                        generated_idea = idea_result.data['generated_idea']
                        sources_used = idea_result.data.get('sources_used', 0)
                        total_docs = idea_result.data.get('total_documents', 0)
                        
                        reasoning += f" Generated content based on {sources_used} unique sources from {total_docs} total documents."
                        
                        # Stream the generated idea directly
                        for chunk in generated_idea:
                            yield chunk
                        
                        return  # Exit early since we've already streamed the response
                    else:
                        reasoning += " Failed to generate ideas from knowledge base."
            else:
                reasoning = "No knowledge base tools needed for this query."
            
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
            
            # Decide if knowledge base should be used and which tool
            should_use_kb, tool_name, search_query = self._should_use_knowledge_base(user_prompt)
            
            knowledge_context = ""
            if should_use_kb and tool_name:
                if tool_name == "search_knowledge_base" and search_query:
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
                        
                elif tool_name == "summarize_knowledge_base":
                    reasoning = "Summarizing all content in the knowledge base"
                    logger.info(reasoning)
                    
                    # Summarize knowledge base
                    summary_result = self._summarize_knowledge_base_tool()
                    tools_used.append("summarize_knowledge_base")
                    
                    if summary_result.success and summary_result.data:
                        # Use the summary as context
                        summary_data = summary_result.data
                        knowledge_context = f"Knowledge Base Summary:\n{summary_data['summary']}\n\n"
                        knowledge_context += f"Total Sources: {summary_data['total_sources']}\n"
                        knowledge_context += f"Content Types: {', '.join(summary_data['content_types'])}\n\n"
                        
                        # Add document details
                        knowledge_context += "Document Details:\n"
                        for doc in summary_data['documents']:
                            knowledge_context += f"- {doc['source']} ({doc['content_type']}): {doc['chunk_count']} chunks\n"
                        
                        reasoning += f" Analyzed {summary_data['total_chunks']} chunks from {summary_data['total_sources']} sources."
                    else:
                        reasoning += " Failed to summarize knowledge base."
            else:
                reasoning = "No knowledge base tools needed for this query."
            
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