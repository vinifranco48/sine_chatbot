#!/usr/bin/env python3
"""
Script para verificar modelos suportados pelo FastEmbed.
"""
import sys
import os

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(__file__))

try:
    from fastembed import TextEmbedding
    from fastembed.sparse.bm25 import Bm25

    print("=== MODELOS SUPORTADOS FASTEMBED ===\n")

    # Listar modelos dense suportados
    print("MODELOS DENSE SUPORTADOS:")
    dense_models = TextEmbedding.list_supported_models()
    for i, model in enumerate(dense_models[:10], 1):  # Mostrar apenas os primeiros 10
        print(f"{i:2d}. {model['model']}")
        print(f"    Descrição: {model.get('description', 'N/A')}")
        print(f"    Dimensões: {model.get('dim', 'N/A')}")
        print()

    print(f"Total de modelos dense: {len(dense_models)}")

    # Listar modelos sparse suportados
    print("\n" + "="*50)
    print("MODELOS SPARSE SUPORTADOS:")
    try:
        sparse_models = Bm25.list_supported_models()
        for i, model in enumerate(sparse_models, 1):
            print(f"{i}. {model['model']}")
            print(f"   Descrição: {model.get('description', 'N/A')}")
            print()
        print(f"Total de modelos sparse: {len(sparse_models)}")
    except Exception as e:
        print(f"Erro ao listar modelos sparse: {e}")

    # Sugerir modelos recomendados
    print("\n" + "="*50)
    print("MODELOS RECOMENDADOS:")
    print("Dense:")
    recommended_dense = [
        "BAAI/bge-small-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-base-en-v1.5"
    ]
    for model in recommended_dense:
        print(f"  - {model}")

    print("\nSparse:")
    print("  - Qdrant/bm25")

except ImportError as e:
    print(f"Erro de importação: {e}")
    print("Certifique-se de que o FastEmbed está instalado:")
    print("pip install fastembed")
except Exception as e:
    print(f"Erro: {e}")
    import traceback
    print(f"Detalhes: {traceback.format_exc()}")