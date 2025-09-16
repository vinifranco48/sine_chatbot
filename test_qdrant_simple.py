#!/usr/bin/env python3
"""
Teste simples para verificar conexão com Qdrant e recuperar dados.
"""
import sys
import os
from dotenv import load_dotenv
import json

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(__file__))

# Carregar variáveis de ambiente
load_dotenv()

from qdrant_client import QdrantClient
from src.config.settings import Settings

def test_qdrant_connection():
    """
    Testa a conexão básica com Qdrant.
    """
    print("=== TESTE DE CONEXÃO QDRANT ===\n")

    try:
        # Inicializar configurações
        settings = Settings()
        print(f"Qdrant URL: {settings.qdrant_url}")
        print(f"Collection: {settings.qdrant_collection_name}")

        # Conectar ao Qdrant
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )

        # Verificar se a coleção existe
        collection_exists = client.collection_exists(
            collection_name=settings.qdrant_collection_name
        )

        if collection_exists:
            print(f"[OK] Coleção '{settings.qdrant_collection_name}' encontrada")

            # Obter informações da coleção
            collection_info = client.get_collection(
                collection_name=settings.qdrant_collection_name
            )

            print(f"Total de pontos: {collection_info.points_count}")

            # Tentar fazer uma busca simples (scroll) para ver alguns documentos
            print("\nBuscando documentos de exemplo...")

            result = client.scroll(
                collection_name=settings.qdrant_collection_name,
                limit=3,  # Apenas 3 documentos
                with_payload=True,
                with_vectors=False  # Não precisamos dos vetores para este teste
            )

            points = result[0]

            print(f"Documentos encontrados: {len(points)}")

            for i, point in enumerate(points, 1):
                print(f"\n--- Documento {i} ---")
                payload = point.payload

                # Extrair informações principais
                product_name = payload.get('product_name', 'N/A')
                manufacturer = payload.get('manufacturer', 'N/A')
                product_type = payload.get('product_type', 'N/A')
                text = payload.get('text', '')

                print(f"ID: {point.id}")
                print(f"Produto: {product_name}")
                print(f"Fabricante: {manufacturer}")
                print(f"Tipo: {product_type}")
                print(f"Texto (primeiros 200 chars): {text[:200]}{'...' if len(text) > 200 else ''}")

                # Mostrar outras propriedades do payload
                other_keys = [k for k in payload.keys()
                             if k not in ['product_name', 'manufacturer', 'product_type', 'text']]
                if other_keys:
                    print(f"Outras propriedades: {other_keys}")

        else:
            print(f"[ERRO] Coleção '{settings.qdrant_collection_name}' não encontrada")

            # Listar coleções existentes
            collections = client.get_collections()
            print(f"Coleções disponíveis: {[c.name for c in collections.collections]}")

    except Exception as e:
        print(f"[ERRO] Erro: {str(e)}")
        import traceback
        print(f"Detalhes: {traceback.format_exc()}")

def test_qdrant_search_by_text():
    """
    Testa busca por texto usando busca por similaridade (sem embeddings complexos).
    """
    print("\n=== TESTE DE BUSCA POR TEXTO ===\n")

    try:
        settings = Settings()
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )

        # Verificar se podemos fazer busca por texto
        collection_name = settings.qdrant_collection_name

        # Usar scroll com filtro de texto simples
        print("Buscando documentos que contenham 'fungicida'...")

        # Fazer scroll de todos os documentos e filtrar localmente
        # (não é o mais eficiente, mas funciona para teste)
        offset = None
        found_docs = []
        search_term = "fungicida"

        while len(found_docs) < 5:  # Parar quando encontrar 5 documentos
            result = client.scroll(
                collection_name=collection_name,
                offset=offset,
                limit=50,  # Buscar em lotes de 50
                with_payload=True,
                with_vectors=False
            )

            points, next_offset = result

            if not points:
                break

            # Filtrar documentos que contenham o termo
            for point in points:
                text = point.payload.get('text', '').lower()
                product_name = point.payload.get('product_name', '').lower()

                if search_term.lower() in text or search_term.lower() in product_name:
                    found_docs.append(point)
                    if len(found_docs) >= 5:
                        break

            if next_offset is None:
                break

            offset = next_offset

        print(f"Documentos encontrados com '{search_term}': {len(found_docs)}")

        for i, point in enumerate(found_docs, 1):
            print(f"\n--- Resultado {i} ---")
            payload = point.payload

            product_name = payload.get('product_name', 'N/A')
            manufacturer = payload.get('manufacturer', 'N/A')
            text = payload.get('text', '')

            print(f"Produto: {product_name}")
            print(f"Fabricante: {manufacturer}")
            print(f"Texto (primeiros 300 chars): {text[:300]}{'...' if len(text) > 300 else ''}")

    except Exception as e:
        print(f"[ERRO] Erro na busca: {str(e)}")
        import traceback
        print(f"Detalhes: {traceback.format_exc()}")

if __name__ == "__main__":
    print("TESTES SIMPLES QDRANT")
    print("=" * 50)

    test_qdrant_connection()
    test_qdrant_search_by_text()

    print("\nTESTES CONCLUÍDOS")