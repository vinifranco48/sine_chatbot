#!/usr/bin/env python3
"""
Teste para verificar o funcionamento do Agent e integração com chatbot.
"""

import sys
import os
import asyncio
import json
from typing import Dict, Any
from unittest.mock import Mock, patch

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.agent import create_compiled_graph, Agent
from src.service.embeddings_services import QueryEmbedder
from src.service.retriever_service import QdrantRetriever
from src.service.llm_service import LLMService
from src.models.embeddings import QueryEmbeddings, Document


class TestAgent:
    """Classe para testar o Agent e suas funcionalidades"""

    def __init__(self):
        self.test_results = {
            'embed_query': False,
            'retrieve_documents': False,
            'generate_response': False,
            'full_pipeline': False
        }

    def setup_mocks(self):
        """Configura os mocks necessários para os testes"""
        # Mock QueryEmbedder
        self.mock_query_embedder = Mock(spec=QueryEmbedder)
        mock_embeddings = QueryEmbeddings(
            dense_embedding=[0.1] * 256,
            sparse_embedding={"tokens": [1, 2, 3], "values": [0.5, 0.3, 0.2]},
            late_interaction_embedding=[[0.1, 0.2], [0.3, 0.4]]
        )
        self.mock_query_embedder.embed_query.return_value = mock_embeddings

        # Mock QdrantRetriever
        self.mock_retriever = Mock(spec=QdrantRetriever)
        mock_documents = [
            Document(
                page_content="Este é um documento de teste sobre agricultura.",
                metadata={"source": "test1.pdf", "page": 1}
            ),
            Document(
                page_content="Informações sobre cultivo de milho e fertilizantes.",
                metadata={"source": "test2.pdf", "page": 2}
            )
        ]
        self.mock_retriever.search_documents.return_value = mock_documents

        # Mock LLMService
        self.mock_llm_service = Mock(spec=LLMService)
        self.mock_llm_service.generate_response.return_value = (
            "Baseado nas informações fornecidas, posso ajudar com questões sobre agricultura. "
            "O milho é uma cultura importante que requer cuidados específicos com fertilizantes."
        )

    def test_embed_query_node(self):
        """Testa o nó de embedding da query"""
        print("\n🧪 Testando embed_query_node...")

        try:
            from src.core.agent import embed_query_node

            # Estado inicial
            state = Agent(
                query="Como plantar milho?",
                filters=None,
                query_embedding=None,
                retrieved_docs=[],
                context="",
                response=None,
                error=None
            )

            # Executar o nó
            result = embed_query_node(state, self.mock_query_embedder)

            # Verificar resultado
            if result.get('query_embedding') and not result.get('error'):
                print("   ✅ embed_query_node funcionou corretamente")
                self.test_results['embed_query'] = True
                return True
            else:
                print(f"   ❌ embed_query_node falhou: {result.get('error')}")
                return False

        except Exception as e:
            print(f"   ❌ Erro no teste embed_query_node: {e}")
            return False

    def test_retrieve_documents_node(self):
        """Testa o nó de recuperação de documentos"""
        print("\n🔍 Testando retrieve_documents_node...")

        try:
            from src.core.agent import retrieve_documents_node

            # Estado com embeddings
            mock_embeddings = QueryEmbeddings(
                dense_embedding=[0.1] * 256,
                sparse_embedding={"tokens": [1, 2, 3], "values": [0.5, 0.3, 0.2]},
                late_interaction_embedding=[[0.1, 0.2], [0.3, 0.4]]
            )

            state = Agent(
                query="Como plantar milho?",
                filters=None,
                query_embedding=mock_embeddings,
                retrieved_docs=[],
                context="",
                response=None,
                error=None
            )

            # Executar o nó
            result = retrieve_documents_node(state, self.mock_retriever)

            # Verificar resultado
            if (result.get('retrieved_docs') and
                result.get('context') and
                not result.get('error')):
                print("   ✅ retrieve_documents_node funcionou corretamente")
                print(f"   📄 Documentos recuperados: {len(result['retrieved_docs'])}")
                self.test_results['retrieve_documents'] = True
                return True
            else:
                print(f"   ❌ retrieve_documents_node falhou: {result.get('error')}")
                return False

        except Exception as e:
            print(f"   ❌ Erro no teste retrieve_documents_node: {e}")
            return False

    def test_generate_response_node(self):
        """Testa o nó de geração de resposta"""
        print("\n💬 Testando generate_response_node...")

        try:
            from src.core.agent import generate_response_node

            # Estado com contexto
            state = Agent(
                query="Como plantar milho?",
                filters=None,
                query_embedding=None,
                retrieved_docs=[],
                context="Este é um documento sobre agricultura e cultivo de milho.",
                response=None,
                error=None
            )

            # Executar o nó
            result = generate_response_node(state, self.mock_llm_service)

            # Verificar resultado
            if result.get('response') and not result.get('error'):
                print("   ✅ generate_response_node funcionou corretamente")
                print(f"   💬 Resposta gerada: {result['response'][:100]}...")
                self.test_results['generate_response'] = True
                return True
            else:
                print(f"   ❌ generate_response_node falhou: {result.get('error')}")
                return False

        except Exception as e:
            print(f"   ❌ Erro no teste generate_response_node: {e}")
            return False

    def test_full_pipeline(self):
        """Testa o pipeline completo do agent"""
        print("\n🚀 Testando pipeline completo do agent...")

        try:
            # Criar o grafo compilado
            compiled_graph = create_compiled_graph(
                query_embedder=self.mock_query_embedder,
                qdrant_retriever=self.mock_retriever,
                llm_service=self.mock_llm_service
            )

            # Estado inicial
            initial_state = Agent(
                query="Como plantar milho de forma sustentável?",
                filters=None,
                query_embedding=None,
                retrieved_docs=[],
                context="",
                response=None,
                error=None
            )

            # Executar o grafo
            final_state = compiled_graph.invoke(initial_state)

            # Verificar resultado final
            if (final_state.get('response') and
                not final_state.get('error') and
                final_state.get('context')):
                print("   ✅ Pipeline completo funcionou corretamente!")
                print(f"   💬 Resposta final: {final_state['response'][:150]}...")
                self.test_results['full_pipeline'] = True
                return True
            else:
                print(f"   ❌ Pipeline completo falhou: {final_state.get('error')}")
                return False

        except Exception as e:
            print(f"   ❌ Erro no teste do pipeline completo: {e}")
            return False

    def test_chatbot_integration(self):
        """Testa diferentes tipos de perguntas como um chatbot real"""
        print("\n💭 Testando integração com chatbot (simulação)...")

        test_queries = [
            "Como plantar milho?",
            "Qual é o melhor fertilizante para soja?",
            "Como controlar pragas na agricultura orgânica?",
            "Quando é a melhor época para plantar tomate?",
            "Como fazer compostagem?"
        ]

        successful_queries = 0

        try:
            compiled_graph = create_compiled_graph(
                query_embedder=self.mock_query_embedder,
                qdrant_retriever=self.mock_retriever,
                llm_service=self.mock_llm_service
            )

            for i, query in enumerate(test_queries, 1):
                print(f"   🗨️ Teste {i}: {query}")

                initial_state = Agent(
                    query=query,
                    filters=None,
                    query_embedding=None,
                    retrieved_docs=[],
                    context="",
                    response=None,
                    error=None
                )

                try:
                    result = compiled_graph.invoke(initial_state)

                    if result.get('response') and not result.get('error'):
                        print(f"      ✅ Sucesso! Resposta: {result['response'][:80]}...")
                        successful_queries += 1
                    else:
                        print(f"      ❌ Falhou: {result.get('error', 'Erro desconhecido')}")

                except Exception as e:
                    print(f"      ❌ Exceção: {e}")

            success_rate = successful_queries / len(test_queries)
            print(f"\n   📊 Taxa de sucesso: {successful_queries}/{len(test_queries)} ({success_rate*100:.1f}%)")

            if success_rate >= 0.8:
                print("   ✅ Integração com chatbot funcionando bem!")
                return True
            else:
                print("   ⚠️ Integração com chatbot precisa de ajustes")
                return False

        except Exception as e:
            print(f"   ❌ Erro no teste de integração: {e}")
            return False

    def run_all_tests(self):
        """Executa todos os testes"""
        print("🎯 === TESTANDO AGENT E CHATBOT ===")

        # Setup
        self.setup_mocks()

        # Executar testes individuais
        tests = [
            ("Embedding da Query", self.test_embed_query_node),
            ("Recuperação de Documentos", self.test_retrieve_documents_node),
            ("Geração de Resposta", self.test_generate_response_node),
            ("Pipeline Completo", self.test_full_pipeline),
            ("Integração Chatbot", self.test_chatbot_integration)
        ]

        passed_tests = 0
        total_tests = len(tests)

        for test_name, test_func in tests:
            print(f"\n{'='*50}")
            print(f"🧪 TESTE: {test_name}")
            print(f"{'='*50}")

            if test_func():
                passed_tests += 1

        # Resultado final
        print(f"\n{'='*60}")
        print("🏁 RESULTADO FINAL DOS TESTES")
        print(f"{'='*60}")
        print(f"✅ Testes passou: {passed_tests}/{total_tests}")
        print(f"📊 Taxa de sucesso: {passed_tests/total_tests*100:.1f}%")

        if passed_tests == total_tests:
            print("🎉 TODOS OS TESTES PASSARAM! O Agent está funcionando perfeitamente.")
        elif passed_tests >= total_tests * 0.8:
            print("✅ A maioria dos testes passou. O Agent está funcionando bem com pequenos ajustes necessários.")
        else:
            print("⚠️ Vários testes falharam. O Agent precisa de correções significativas.")

        print("\n📋 Detalhes dos resultados:")
        for component, status in self.test_results.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component}")

        return passed_tests / total_tests


def main():
    """Função principal para executar os testes"""
    tester = TestAgent()
    success_rate = tester.run_all_tests()

    # Exit code baseado no resultado
    exit_code = 0 if success_rate >= 0.8 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()