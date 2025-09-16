# test_agent.py

import pprint
from typing import Dict, Any

# Importa a classe de configurações
from src.config.settings import settings

# Importa o construtor do grafo
from src.core.agent import create_compiled_graph 

# Importa os serviços necessários para a inicialização
from src.service.embeddings_services import QueryEmbedder
from src.service.retriever_service import QdrantRetriever # (Você precisa criar este arquivo)
from src.service.llm_service import LLMService             # (Você precisa criar este arquivo)

# Mock das classes de serviço que ainda não foram implementadas
# Remova isso quando tiver os arquivos reais
if not hasattr(__import__('src.service.retriever_service'), 'QdrantRetriever'):
    class QdrantRetriever:
        def search_documents(self, embeddings, limit):
            print("AVISO: Usando QdrantRetriever mockado. Nenhum documento real será retornado.")
            return []

if not hasattr(__import__('src.service.llm_service'), 'LLMService'):
    class LLMService:
        def generate_response(self, prompt):
            print("AVISO: Usando LLMService mockado. Retornando resposta padrão.")
            return "Esta é uma resposta padrão do LLM mockado, pois o contexto estava vazio."


def run_agro_assistant_test(query: str) -> Dict[str, Any]:
    """
    Função principal para inicializar, compilar e executar o agente AgroAssistente.
    """
    print("--- INICIANDO TESTE DO AGROASSISTENTE ---")

    # 1. Inicialização dos Serviços com base nas Configurações
    print("1. Inicializando os serviços (Embedder, Retriever, LLM)...")
    try:
        # **A CORREÇÃO ESTÁ AQUI**
        # Inicializa o QueryEmbedder com os nomes dos modelos das configurações
        query_embedder = QueryEmbedder(
            dense_model_name=settings.dense_model_name,
            sparse_model_name=settings.bm25_model_name

        )
        
        # Outros serviços também usariam 'settings' para suas configs
        qdrant_retriever = QdrantRetriever()
        llm_service = LLMService()
        print("   Serviços inicializados com sucesso.")

    except Exception as e:
        print(f"   ERRO ao inicializar serviços: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Falha na inicialização: {e}"}

    # 2. Criação e Compilação do Grafo
    print("\n2. Compilando o grafo LangGraph do agente...")
    agro_assistant_app = create_compiled_graph(
        query_embedder=query_embedder,
        qdrant_retriever=qdrant_retriever,
        llm_service=llm_service
    )
    print("   Grafo compilado.")

    # 3. Definição do Estado Inicial
    initial_state = {"query": query}
    print(f"\n3. Query de entrada: '{query}'")

    # 4. Execução do Agente
    print("\n4. Executando o fluxo do agente...")
    final_state = agro_assistant_app.invoke(initial_state)
    print("   Execução concluída.")
    
    return final_state

if __name__ == "__main__":
    user_query = "qual melhor produto para plantas daninhas "
    result = run_agro_assistant_test(user_query)

    print("\n--- RESULTADO FINAL (ESTADO COMPLETO DO AGENTE) ---")
    pprint.pprint(result)

    print("\n--- RESPOSTA GERADA ---")
    if result.get("error"):
        print(f"Ocorreu um erro durante a execução: {result.get('error')}")
    elif result.get("response"):
        print(result["response"])
    else:
        print("Nenhuma resposta ou erro foi retornado no estado final.")