from src.models.embeddings import QueryEmbeddings, Document 
from src.service.embeddings_services import QueryEmbedder 
from src.service.retriever_service import QdrantRetriever 
from src.service.llm_service import LLMService
from src.service.prompt_service import format_rag_prompt, get_disclaimer, should_include_disclaimer
from src.config.settings import settings
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
import json
import traceback
import time


class Agent(TypedDict):
    """Defines the state that flows through the graph for AgroAssistente."""
    query: str
    filters: Optional[Dict[str, Any]]
    query_embedding: Optional[QueryEmbeddings] 
    retrieved_docs: List[Document] 
    context: str
    response: Optional[str]
    error: Optional[str]


def embed_query_node(state: Agent, query_embedder: QueryEmbedder) -> Dict[str, Any]:
    """
    Node to generate hybrid embeddings (dense, sparse, late) for the user's query.
    """
    print('--- Node: Embed Query (Hybrid) ---')
    query = state.get("query")
    if not query:
        error_msg = "Query not found in state."
        print(f"Error: {error_msg}")
        return {"error": json.dumps({"node": "embed_query", "message": error_msg})}

    try:
        print(f"Generating hybrid embeddings for query: '{query[:100]}...'")
        start_time = time.time()
        embeddings = query_embedder.embed_query(query)
        end_time = time.time()
        print(f"Embeddings generated in {end_time - start_time:.4f}s.")

        if not embeddings:
            raise ValueError("Failed to generate embeddings (empty result).")

        return {"query_embedding": embeddings, "error": None}

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Exception in embed_query_node: {e}\n{error_details}")
        return {"error": json.dumps({"node": "embed_query", "message": str(e), "details": error_details})}


def retrieve_documents_node(state: Agent, qdrant_retriever: QdrantRetriever) -> Dict[str, Any]:
    """
    Node to retrieve documents using hybrid search (prefetch) and reranking (late-interaction).
    """
    print('--- Node: Retrieve Documents (Hybrid) ---')
    if state.get("error"):
        print(f"Previous error detected, skipping retrieval: {state.get('error')}")
        return {"retrieved_docs": [], "context": ""}

    query_embedding = state.get("query_embedding")
    if not query_embedding:
        error_msg = "Embeddings object not found in state."
        print(f"Error: {error_msg}")
        return {"retrieved_docs": [], "context": "", "error": json.dumps({"node": "retrieve_documents", "message": error_msg})}

    try:
        print("Searching documents with hybrid search and reranking...")
        retrieved_documents = qdrant_retriever.search_documents(
            embeddings=query_embedding,
            limit=settings.retrieval_limit
        )
        print(f"Retrieved {len(retrieved_documents)} documents after reranking.")

        context_texts = [doc.page_content for doc in retrieved_documents if doc.page_content]

        if not context_texts:
            print("Warning: No text found in retrieved documents.")
            context = ""
        else:
            context = "\n\n---\n\n".join(context_texts)
            print(f"Context assembled (first 200 chars): {context[:200]}...")

        return {
            "retrieved_docs": retrieved_documents,
            "context": context,
            "error": None
        }

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Exception while retrieving documents: {e}\n{error_details}")
        return {"retrieved_docs": [], "context": "", "error": json.dumps({"node": "retrieve_documents", "message": str(e), "details": error_details})}


def generate_response_node(state: Agent, llm_service: LLMService) -> Dict[str, Any]:
    """Node to generate the final response using the LLM with the AgroAssistente prompt."""
    print('--- Node: Generate Response ---')
    if state.get('error'):
        print(f"Previous error detected, skipping response generation: {state.get('error')}")
        error_info = json.loads(state.get('error'))
        return {"response": f"Sorry, an error occurred in step '{error_info.get('node', 'unknown')}': {error_info.get('message', 'Internal error.')}"}

    query = state.get('query')
    context = state.get('context', "")
    
    try:
        prompt = format_rag_prompt(query=query, context=context)
        print(f"DEBUG: Prompt for LLM (start): {prompt[:200]}...")

        response_text = llm_service.generate_response(prompt)
        if not response_text:
            raise ValueError("Failed to generate LLM response (empty response).")

        final_response = response_text
        if should_include_disclaimer(response_text):
            final_response += get_disclaimer()

        print(f"Final response generated (start): {final_response[:200]}...")
        return {"response": final_response, "error": None}

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Exception while generating response with LLM: {e}\n{error_details}")
        return {"response": None, "error": json.dumps({"node": "generate_response", "message": str(e), "details": error_details})}


def create_compiled_graph(
    query_embedder: QueryEmbedder,
    qdrant_retriever: QdrantRetriever,
    llm_service: LLMService
):
    """Builds and compiles the LangGraph graph for the AgroAssistente flow."""
    print("Building and compiling the LangGraph graph for AgroAssistente...")

    workflow = StateGraph(Agent)
    workflow.add_node("embed_query", lambda state: embed_query_node(state, query_embedder))
    workflow.add_node("retrieve_documents", lambda state: retrieve_documents_node(state, qdrant_retriever))
    workflow.add_node("generate_response", lambda state: generate_response_node(state, llm_service))

    workflow.set_entry_point("embed_query")
    workflow.add_edge("embed_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_response")
    workflow.add_edge("generate_response", END)
    
    compiled_graph = workflow.compile()
    print("AgroAssistente graph compiled successfully.")
    return compiled_graph