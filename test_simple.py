# -*- coding: utf-8 -*-
"""
Teste simples para verificar funcionamento do Agent
"""

import sys
import os
from src.core.agent import create_compiled_graph, Agent
from src.service.embeddings_services import QueryEmbedder
from src.service.retriever_service import QdrantRetriever
from src.service.llm_service import LLMService
from src.config.settings import settings

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Testa se todos os imports funcionam"""
    try:
        from src.core.agent import create_compiled_graph, Agent
        from src.service.embeddings_services import QueryEmbedder
        from src.service.retriever_service import QdrantRetriever
        from src.service.llm_service import LLMService
        from src.config.settings import settings
        print("OK - Todos os imports funcionaram")
        return True
    except Exception as e:
        print(f"ERRO - Import falhou: {e}")
        return False

def test_services():
    """Testa inicializacao dos servicos"""
    try:
        print("Inicializando servicos...")

        query_embedder = QueryEmbedder(
            dense_model_name=settings.bedrock_embedding_model, 
            sparse_model_name=settings.bm25_model_name,
            aws_region=settings.aws_region
        )
        print("QueryEmbedder: OK")

        qdrant_retriever = QdrantRetriever(settings=settings)
        print("QdrantRetriever: OK")

        llm_service = LLMService(settings=settings)
        print("LLMService: OK")

        return True, (query_embedder, qdrant_retriever, llm_service)
    except Exception as e:
        print(f"ERRO - Servicos falharam: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_agent_pipeline(services):
    """Testa o pipeline completo do agent"""
    query_embedder, qdrant_retriever, llm_service = services

    try:
        print("Criando grafo compilado...")
        compiled_graph = create_compiled_graph(
            query_embedder=query_embedder,
            qdrant_retriever=qdrant_retriever,
            llm_service=llm_service
        )
        print("Grafo criado: OK")

        print("Testando query: 'Como plantar milho?'")
        initial_state = {
            'query': "Como plantar milho?",
            'filters': None,
            'query_embedding': None,
            'retrieved_docs': [],
            'context': "",
            'response': None,
            'error': None
        }

        result = compiled_graph.invoke(initial_state)

        if result.get('response') and not result.get('error'):
            print("SUCESSO! Agent funcionou")
            print(f"Resposta: {result['response'][:100]}...")
            return True
        else:
            print("FALHA! Agent nao gerou resposta")
            if result.get('error'):
                print(f"Erro: {result['error']}")
            return False

    except Exception as e:
        print(f"ERRO - Pipeline falhou: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Funcao principal"""
    print("=== TESTE SIMPLES DO AGENT ===")

    # Teste 1: Imports
    print("\n1. Testando imports...")
    if not test_imports():
        print("FALHOU nos imports!")
        return

    # Teste 2: Servicos
    print("\n2. Testando servicos...")
    success, services = test_services()
    if not success:
        print("FALHOU nos servicos!")
        return

    # Teste 3: Pipeline
    print("\n3. Testando pipeline completo...")
    if test_agent_pipeline(services):
        print("\n=== RESULTADO: CHATBOT FUNCIONANDO! ===")
    else:
        print("\n=== RESULTADO: CHATBOT COM PROBLEMAS ===")

if __name__ == "__main__":
    main()