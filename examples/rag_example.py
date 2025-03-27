"""
Retrieval-Augmented Generation (RAG) Example

This script demonstrates how to use the vector store integration and RAG capabilities
to enhance an agent with external knowledge.
"""

import asyncio
import logging
import os
import sys
import argparse
from typing import List, Dict, Any

# Add parent directory to path to run as standalone script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_agent_framework.core.llm.factory import LLMFactory
from ai_agent_framework.agents.workflow_agent import WorkflowAgent
from ai_agent_framework.core.tools.registry import ToolRegistry
from ai_agent_framework.tools.memory.retrieval_tool import RetrievalTool
from ai_agent_framework.core.memory.embeddings import get_embedder
from ai_agent_framework.core.memory.text_utils import chunk_text, MarkdownTextSplitter
from ai_agent_framework.core.memory.vector_store import Document
from ai_agent_framework.config.logging_config import setup_logging

# Set up logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


async def load_documents(
    directory: str,
    embedder: Any,
    retrieval_tool: RetrievalTool
) -> int:
    """
    Load documents from a directory into the knowledge base.
    
    Args:
        directory: Directory containing documents
        embedder: Embedder to use for creating embeddings
        retrieval_tool: Retrieval tool to add documents to
        
    Returns:
        Number of documents loaded
    """
    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        return 0
        
    loaded_count = 0
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Walk through directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Only process text files
            if file.endswith(('.md', '.txt', '.rst')):
                file_path = os.path.join(root, file)
                
                try:
                    # Read file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Determine format
                    text_format = "markdown" if file.endswith('.md') else "text"
                    
                    # Create document metadata
                    metadata = {
                        "source": file_path,
                        "filename": file,
                        "format": text_format
                    }
                    
                    # Split into chunks
                    chunks = splitter.split_text(content)
                    
                    # Create documents
                    documents = []
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            text=chunk,
                            metadata={
                                **metadata,
                                "chunk": i,
                                "chunk_count": len(chunks)
                            }
                        )
                        documents.append(doc)
                    
                    # Add to knowledge base
                    await retrieval_tool.add_documents(documents)
                    loaded_count += len(documents)
                    
                    logger.info(f"Loaded {len(documents)} chunks from {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
    
    return loaded_count


async def run_rag_example(knowledge_dir: str, vector_store_path: str = ".vector_store"):
    """
    Run the RAG example.
    
    Args:
        knowledge_dir: Directory containing knowledge documents
        vector_store_path: Path to store vector data
    """
    # Initialize embedder
    embedder = get_embedder(embedder_type="openai")
    
    # Create retrieval tool
    retrieval_tool = RetrievalTool(
        vector_store_type="chroma",
        vector_store_path=vector_store_path,
        collection_name="knowledge_base",
        embedder=embedder,
        default_k=3
    )
    
    # Load documents
    doc_count = await load_documents(knowledge_dir, embedder, retrieval_tool)
    logger.info(f"Loaded {doc_count} document chunks into knowledge base")
    
    # Get knowledge base stats
    stats = await retrieval_tool.get_stats()
    logger.info(f"Knowledge base stats: {stats}")
    
    # Create LLM
    llm = LLMFactory.create_llm(provider="claude")
    
    # Set up tool registry
    tools = ToolRegistry()
    tools.register_tool(retrieval_tool)
    
    # Create the agent
    agent = WorkflowAgent(
        name="rag_agent",
        llm=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful assistant with access to a knowledge base. "
            "When asked a question, search the knowledge base using the retrieve tool "
            "to find relevant information before responding. "
            "Always cite your sources by mentioning the filename the information came from."
        )
    )
    
    # Interactive chat loop
    print("\n===== RAG-enhanced Agent =====")
    print("(Type 'exit' to quit)\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            # Process with agent
            print("\nAgent is thinking...")
            response = await agent.run(user_input)
            
            # Display response
            print(f"\nAgent: {response['response']}\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


async def run_retrieval_example(knowledge_dir: str, vector_store_path: str = ".vector_store"):
    """
    Run just the retrieval component to test it.
    
    Args:
        knowledge_dir: Directory containing knowledge documents
        vector_store_path: Path to store vector data
    """
    # Initialize embedder
    embedder = get_embedder(embedder_type="openai")
    
    # Create retrieval tool
    retrieval_tool = RetrievalTool(
        vector_store_type="chroma",
        vector_store_path=vector_store_path,
        collection_name="knowledge_base",
        embedder=embedder,
        default_k=5
    )
    
    # Load documents if needed
    stats = await retrieval_tool.get_stats()
    if stats.get("count", 0) == 0:
        logger.info("Knowledge base is empty, loading documents...")
        doc_count = await load_documents(knowledge_dir, embedder, retrieval_tool)
        logger.info(f"Loaded {doc_count} document chunks into knowledge base")
        stats = await retrieval_tool.get_stats()
    
    logger.info(f"Knowledge base stats: {stats}")
    
    # Interactive retrieval loop
    print("\n===== Retrieval Testing =====")
    print("(Type 'exit' to quit)\n")
    
    while True:
        try:
            # Get user input
            user_input = input("Query: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            # Perform retrieval
            print("\nSearching...")
            result = await retrieval_tool.execute(query=user_input, k=3)
            
            # Display results
            print(f"\nFound {len(result['results'])} relevant documents:\n")
            
            for i, doc in enumerate(result["results"], 1):
                print(f"{i}. Score: {doc['score']:.4f}")
                print(f"   Source: {doc.get('metadata', {}).get('source', 'unknown')}")
                print(f"   Content: {doc['content'][:200]}...")
                print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


async def main():
    """Main function to parse args and run examples."""
    parser = argparse.ArgumentParser(description="RAG Example")
    parser.add_argument("--knowledge-dir", type=str, default="./docs",
                        help="Directory containing knowledge documents")
    parser.add_argument("--vector-store-path", type=str, default="./.vector_store",
                        help="Path to store vector data")
    parser.add_argument("--mode", type=str, choices=["rag", "retrieval"], default="rag",
                        help="Mode to run (rag or retrieval)")
    
    args = parser.parse_args()
    
    if args.mode == "rag":
        await run_rag_example(args.knowledge_dir, args.vector_store_path)
    else:
        await run_retrieval_example(args.knowledge_dir, args.vector_store_path)


if __name__ == "__main__":
    asyncio.run(main())