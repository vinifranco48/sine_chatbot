#!/usr/bin/env python3
"""
Teste direto para verificar o funcionamento real do Agent e chatbot.
"""

import sys
import os
import asyncio
from typing import Dict, Any

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.core.agent import create_compiled_graph, Agent
    from src.service.embeddings_services import QueryEmbedder
    from src.service.retriever_service import QdrantRetriever
    from src.service.llm_service import LLMService
    from src.config.settings import settings
    print("✅ Imports realizados com sucesso")
except ImportError as e:
    print(f"❌ Erro de import: {e}")
    sys.exit(1)


def test_agent_real():
    """Testa o agent com os serviços reais"""
    print("🚀 === TESTANDO AGENT COM SERVIÇOS REAIS ===\n")

    try:
        # 1. Inicializar serviços reais
        print("🔧 Inicializando serviços...")

        query_embedder = QueryEmbedder()
        print("   ✅ QueryEmbedder inicializado")

        qdrant_retriever = QdrantRetriever()
        print("   ✅ QdrantRetriever inicializado")

        llm_service = LLMService()
        print("   ✅ LLMService inicializado")

        # 2. Criar o grafo compilado
        print("\n🔗 Criando grafo compilado...")
        compiled_graph = create_compiled_graph(
            query_embedder=query_embedder,
            qdrant_retriever=qdrant_retriever,
            llm_service=llm_service
        )
        print("   ✅ Grafo compilado com sucesso")

        # 3. Testar com diferentes tipos de perguntas
        test_queries = [
            "Como plantar milho?",
            "Qual é o melhor fertilizante para soja?",
            "Como controlar pragas na agricultura?",
            "O que é agricultura sustentável?",
            "Como fazer irrigação eficiente?"
        ]

        successful_tests = 0
        total_tests = len(test_queries)

        print(f"\n🧪 Executando {total_tests} testes com queries reais...")
        print("="*60)

        for i, query in enumerate(test_queries, 1):
            print(f"\n🗨️ TESTE {i}/{total_tests}: {query}")
            print("-" * 50)

            try:
                # Estado inicial
                initial_state = Agent(
                    query=query,
                    filters=None,
                    query_embedding=None,
                    retrieved_docs=[],
                    context="",
                    response=None,
                    error=None
                )

                # Executar o grafo
                print("   ⏳ Processando...")
                result = compiled_graph.invoke(initial_state)

                # Verificar resultado
                if result.get('response') and not result.get('error'):
                    print("   ✅ SUCESSO!")
                    print(f"   📄 Docs encontrados: {len(result.get('retrieved_docs', []))}")
                    print(f"   💬 Resposta: {result['response'][:150]}...")
                    successful_tests += 1
                else:
                    print("   ❌ FALHA!")
                    if result.get('error'):
                        print(f"   🔍 Erro: {result['error']}")
                    else:
                        print("   🔍 Nenhuma resposta gerada")

            except Exception as e:
                print(f"   ❌ EXCEÇÃO: {str(e)[:100]}...")
                import traceback
                traceback.print_exc()

        # 4. Resultado final
        print("\n" + "="*60)
        print("🏁 RESULTADO FINAL")
        print("="*60)

        success_rate = successful_tests / total_tests
        print(f"✅ Testes bem-sucedidos: {successful_tests}/{total_tests}")
        print(f"📊 Taxa de sucesso: {success_rate*100:.1f}%")

        if success_rate == 1.0:
            print("🎉 PERFEITO! Todos os testes passaram!")
            status = "EXCELENTE"
        elif success_rate >= 0.8:
            print("✅ MUITO BOM! A maioria dos testes passou!")
            status = "BOM"
        elif success_rate >= 0.6:
            print("⚠️ RAZOÁVEL. Alguns problemas detectados.")
            status = "REGULAR"
        else:
            print("❌ PROBLEMÁTICO. Muitos testes falharam.")
            status = "RUIM"

        print(f"\n🎯 STATUS DO AGENT: {status}")

        return success_rate

    except Exception as e:
        print(f"❌ ERRO CRÍTICO ao inicializar serviços: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def test_individual_components():
    """Testa componentes individuais para diagnóstico"""
    print("\n🔍 === TESTANDO COMPONENTES INDIVIDUAIS ===")

    results = {
        'embedder': False,
        'retriever': False,
        'llm': False
    }

    # Testar QueryEmbedder
    try:
        print("\n🧠 Testando QueryEmbedder...")
        embedder = QueryEmbedder()
        embeddings = embedder.embed_query("teste")
        if embeddings and embeddings.dense_embedding:
            print("   ✅ QueryEmbedder funcionando")
            results['embedder'] = True
        else:
            print("   ❌ QueryEmbedder retornou embeddings vazios")
    except Exception as e:
        print(f"   ❌ Erro no QueryEmbedder: {e}")

    # Testar QdrantRetriever
    try:
        print("\n📚 Testando QdrantRetriever...")
        retriever = QdrantRetriever()
        # Aqui precisaríamos de embeddings para testar, então vamos só verificar a inicialização
        print("   ✅ QdrantRetriever inicializado")
        results['retriever'] = True
    except Exception as e:
        print(f"   ❌ Erro no QdrantRetriever: {e}")

    # Testar LLMService
    try:
        print("\n🤖 Testando LLMService...")
        llm = LLMService()
        response = llm.generate_response("Diga olá")
        if response:
            print(f"   ✅ LLMService funcionando: {response[:50]}...")
            results['llm'] = True
        else:
            print("   ❌ LLMService retornou resposta vazia")
    except Exception as e:
        print(f"   ❌ Erro no LLMService: {e}")

    return results


def main():
    """Função principal"""
    print("🎯 TESTE COMPLETO DO AGENT SINE CHATBOT")
    print("=" * 50)

    # Testar componentes individuais primeiro
    component_results = test_individual_components()

    # Se componentes básicos funcionam, testar o agent completo
    if any(component_results.values()):
        agent_success_rate = test_agent_real()
    else:
        print("\n❌ Componentes básicos falharam. Não é possível testar o agent completo.")
        agent_success_rate = 0.0

    # Resumo final
    print("\n" + "=" * 60)
    print("📋 RESUMO FINAL")
    print("=" * 60)
    print("Componentes:")
    for component, status in component_results.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {component}")

    print(f"\nAgent completo: {agent_success_rate*100:.1f}% de sucesso")

    if agent_success_rate >= 0.8 and all(component_results.values()):
        print("\n🎉 CHATBOT ESTÁ FUNCIONANDO CORRETAMENTE!")
    elif agent_success_rate >= 0.6:
        print("\n✅ CHATBOT ESTÁ FUNCIONANDO COM ALGUNS PROBLEMAS MENORES")
    else:
        print("\n⚠️ CHATBOT PRECISA DE CORREÇÕES")

    return agent_success_rate


if __name__ == "__main__":
    main()