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
    """Define o estado que flui através do grafo para o AgroAssistente."""
    query: str
    filters: Optional[Dict[str, Any]]
    query_embedding: Optional[QueryEmbeddings] 
    retrieved_docs: List[Document] 
    context: str
    response: Optional[str]
    error: Optional[str]


def embed_query_node(state: Agent, query_embedder: QueryEmbedder) -> Dict[str, Any]:
    """
    Nó para gerar os embeddings híbridos (dense, sparse, late) da query do usuário.
    """
    print('--- Nó: Embed Query (Híbrido) ---')
    query = state.get("query")
    if not query:
        error_msg = "Query não encontrada no estado."
        print(f"Erro: {error_msg}")
        return {"error": json.dumps({"node": "embed_query", "message": error_msg})}

    try:
        print(f"Gerando embeddings híbridos para a query: '{query[:100]}...'")
        start_time = time.time()
        # MUDANÇA: Usa o serviço QueryEmbedder para obter todos os embeddings de uma vez
        embeddings = query_embedder.embed_query(query)
        end_time = time.time()
        print(f"Embeddings gerados em {end_time - start_time:.4f}s.")

        if not embeddings:
            raise ValueError("Falha ao gerar embeddings (resultado vazio).")

        return {"query_embedding": embeddings, "error": None}

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Erro excepcional no embed_query_node: {e}\n{error_details}")
        return {"error": json.dumps({"node": "embed_query", "message": str(e), "details": error_details})}


def retrieve_documents_node(state: Agent, qdrant_retriever: QdrantRetriever) -> Dict[str, Any]:
    """
    Nó para recuperar documentos usando busca híbrida (prefetch) e reranking (late-interaction).
    """
    print('--- Nó: Retrieve Documents (Híbrido) ---')
    if state.get("error"):
        print(f"Erro anterior detectado, pulando recuperação: {state.get('error')}")
        return {"retrieved_docs": [], "context": ""}

    query_embedding = state.get("query_embedding")
    if not query_embedding:
        error_msg = "Objeto de embeddings não encontrado no estado."
        print(f"Erro: {error_msg}")
        return {"retrieved_docs": [], "context": "", "error": json.dumps({"node": "retrieve_documents", "message": error_msg})}

    try:
        print("Buscando documentos com busca híbrida e reranking...")
        # MUDANÇA: Usa o método search_documents do QdrantRetriever, que orquestra a busca complexa
        retrieved_documents = qdrant_retriever.search_documents(
            embeddings=query_embedding,
            limit=settings.retrieval_limit
        )
        print(f"Recuperados {len(retrieved_documents)} documentos após reranking.")

        # MUDANÇA: Extrai o contexto dos objetos Document
        context_texts = [doc.page_content for doc in retrieved_documents if doc.page_content]

        if not context_texts:
            print("Aviso: Nenhum texto encontrado nos documentos recuperados.")
            context = ""
        else:
            context = "\n\n---\n\n".join(context_texts)
            print(f"Contexto montado (primeiros 200 chars): {context[:200]}...")

        return {
            "retrieved_docs": retrieved_documents,
            "context": context,
            "error": None
        }

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Erro excepcional ao recuperar documentos: {e}\n{error_details}")
        return {"retrieved_docs": [], "context": "", "error": json.dumps({"node": "retrieve_documents", "message": str(e), "details": error_details})}


def generate_response_node(state: Agent, llm_service: LLMService) -> Dict[str, Any]:
    """Nó para gerar a resposta final usando o LLM com o prompt do AgroAssistente."""
    print('--- Nó: Generate Response ---')
    if state.get('error'):
        print(f"Erro anterior detectado, pulando geração de resposta: {state.get('error')}")
        error_info = json.loads(state.get('error'))
        return {"response": f"Desculpe, ocorreu um erro no passo '{error_info.get('node', 'desconhecido')}': {error_info.get('message', 'Erro interno.')}"}

    query = state.get('query')
    context = state.get('context', "")
    
    try:
        prompt = format_rag_prompt(query=query, context=context)
        print(f"DEBUG: Prompt para LLM (início): {prompt[:200]}...")

        response_text = llm_service.generate_response(prompt)
        if not response_text:
            raise ValueError("Falha ao gerar resposta do LLM (resposta vazia).")

        # MUDANÇA: Adiciona o disclaimer condicionalmente
        final_response = response_text
        if should_include_disclaimer(response_text):
            final_response += get_disclaimer()

        print(f"Resposta final gerada (início): {final_response[:200]}...")
        return {"response": final_response, "error": None}

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Erro excepcional ao gerar resposta com LLM: {e}\n{error_details}")
        return {"response": None, "error": json.dumps({"node": "generate_response", "message": str(e), "details": error_details})}


def create_compiled_graph(
    query_embedder: QueryEmbedder,
    qdrant_retriever: QdrantRetriever,
    llm_service: LLMService
):
    """Constrói e compila o grafo LangGraph para o fluxo do AgroAssistente."""
    print("Construindo e compilando o grafo LangGraph do AgroAssistente...")

    workflow = StateGraph(Agent)

    # Adiciona nós, passando os novos serviços como argumentos
    workflow.add_node("embed_query", lambda state: embed_query_node(state, query_embedder))
    workflow.add_node("retrieve_documents", lambda state: retrieve_documents_node(state, qdrant_retriever))
    workflow.add_node("generate_response", lambda state: generate_response_node(state, llm_service))

    # O fluxo do grafo permanece o mesmo
    workflow.set_entry_point("embed_query")
    workflow.add_edge("embed_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_response")
    workflow.add_edge("generate_response", END)
    
    compiled_graph = workflow.compile()
    print("Grafo do AgroAssistente compilado com sucesso.")
    return compiled_graph