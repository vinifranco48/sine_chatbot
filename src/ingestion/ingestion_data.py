import json
import os
import uuid
from aiohttp import ClientError
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Dict
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, SparseVector, VectorParams, Distance, SparseVectorParams
from fastembed.late_interaction import LateInteractionTextEmbedding
from fastembed import TextEmbedding
from src.config.settings import Settings
import boto3
from fastembed.sparse.bm25 import Bm25


load_dotenv()   
settings = Settings()
DATA_PATH = settings.data_path
EMBED_MODEL = settings.dense_model_name
MAX_TOKENS = 750

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""

    try:
        df = pd.read_excel(file_path, sheet_name=0)
        print(f"Data loaded successfully with {len(df)} records.")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def create_product_chunks(df: pd.DataFrame) -> List[Dict]:
    """Create chunks of product data from the DataFrame."""
    chunks = []

    for index, row in df.iterrows():
        product_text = create_product_text(row)
        metadata = create_product_metadata(row, index)
        chunks.append({"text": product_text, "metadata": metadata})

    return chunks

def create_product_text(row: pd.Series) -> str:
    """Create a text representation of the product."""
    mapeamento_campos = {
        'PRODUTO': 'Produto',
        'TIPO': 'Tipo',
        'EMPRESA (FABRICANTE)': 'Fabricante',
        'ALVO': 'Alvo',
        'APLICAÇÃO': 'Aplicação',
        'DOSE': 'Dose recomendada',
        'CARACTERÍSTICAS E BENEFÍCIOS': 'Características e Benefícios'
    }
    
    text_parts = []
    for coluna, rotulo in mapeamento_campos.items():
        valor = row.get(coluna)
        if pd.notna(valor):
            text_parts.append(f"{rotulo}: {valor}")

    return "\n".join(text_parts)

def create_product_metadata(row: pd.Series, index: int) -> Dict:
    """Create metadata for the product."""
    metadata = {
        "chunk_id": index,
        "product_name": row.get('PRODUTO', '').strip() if pd.notna(row.get('PRODUTO')) else None,
        "product_type": row.get('TIPO', '').strip() if pd.notna(row.get('TIPO')) else None,
        "manufacturer": row.get('EMPRESA (FABRICANTE)', '').strip() if pd.notna(row.get('EMPRESA (FABRICANTE)')) else None,
        "target": row.get('ALVO', '').strip() if pd.notna(row.get('ALVO')) else None,
        "application": row.get('APLICAÇÃO', '').strip() if pd.notna(row.get('APLICAÇÃO')) else None,
        "dosage": row.get('DOSE', '').strip() if pd.notna(row.get('DOSE')) else None,
    }
    
    metadata = {k: v for k, v in metadata.items() if v is not None and v != ''}
    
    return metadata

def initialize_embedding_models():
    """ Initialize the three embedding models needed for hybrid search: """
    print("Initializing embedding models...")
    try:
        bedrock_client = boto3.client(
            service_name="bedrock-runtime", 
            region_name='us-east-2',
        )
        print("AWS Bedrock client initialized successfully.")
    except Exception as e:
        print(f"Error initializing AWS Bedrock client: {e}")
        raise
    bm25_model = Bm25(settings.bm25_model_name)
    print("Models initialized successfully.")

    return bedrock_client, bm25_model

def create_embeddings(chunk_text, bedrock_client, bm25_model):
    """Create embeddings for the given chunk of text using the specified models."""
    try:
        # Corrigir o corpo da requisição para o Titan Text Embeddings V2
        body = json.dumps({
            "inputText": chunk_text,
            "dimensions": 1024,  # Titan V2 suporta até 1024 dimensões
            "normalize": True    # Normalizar os embeddings
        })
        
        # Invoca o modelo na AWS
        response = bedrock_client.invoke_model(
            body=body,
            modelId=EMBED_MODEL,  # "amazon.titan-embed-text-v2:0"
            accept="application/json",
            contentType="application/json"
        )
        
        # Lê e extrai o vetor de embedding da resposta
        response_body = json.loads(response.get("body").read())
        dense_embeddings = response_body.get("embedding")
        
        # Corrigir o acesso ao BM25 (resolver o erro do generator)
        bm25_generator = bm25_model.passage_embed([chunk_text])
        bm25_embeddings = list(bm25_generator)[0]
        
        # Retornar ambos os embeddings
        return {
            'dense': dense_embeddings,
            'sparse': bm25_embeddings
        }
        
    except Exception as e:
        print(f"Erro ao criar embeddings: {type(e).__name__}: {e}")
        print(f"Chunk text: {chunk_text[:100]}...")  # Primeiros 100 caracteres para debug
        raise e

    # A lógica para o BM25 continua a mesma
    bm25_embeddings = bm25_model.passage_embed([chunk_text])[0]
    
    return {
        "dense": dense_embeddings,
        "sparse": bm25_embeddings
    }

def prepare_point(chunk, embedding_models):
    """Prepara um ponto para ser inserido no banco de dados vetorial."""
    
    bedrock_client, bm25_model = embedding_models
    
    text = chunk.get("text", "")

    # --- CORREÇÃO ADICIONADA AQUI ---
    # Verifica se o texto do chunk está vazio ou contém apenas espaços
    if not text or not text.strip():
        print(f"Pulando chunk vazio ou inválido. Metadados: {chunk.get('metadata')}")
        return None
    # --- FIM DA CORREÇÃO ---
    
    embeddings = create_embeddings(text, bedrock_client, bm25_model)
    
    # Validação para pular o chunk se a criação do embedding falhar
    if not embeddings or not embeddings.get("dense"):
        print(f"Pulando chunk por erro no embedding: {chunk['metadata'].get('product_name')}")
        return None

    # Converte o objeto SparseEmbedding (do fastembed) para o SparseVector (do Qdrant)
    sparse_vector_qdrant = SparseVector(
        indices=embeddings["sparse"].indices.tolist(),
        values=embeddings["sparse"].values.tolist()
    )
    
    # Desempacota os metadados para um payload mais plano e fácil de consultar
    payload = {"text": text, **chunk.get("metadata", {})}
    
    return PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": embeddings["dense"],
            "sparse": sparse_vector_qdrant
        },
        payload=payload,
    )

def upload_in_batches(
    client: QdrantClient,
    collection_name: str,
    points: List[PointStruct],
    batch_size: int = 10,
):
    """Upload points to Qdrant in batches."""

    n_batches = (len(points) + batch_size - 1) // batch_size
    print(f"Uploading {len(points)} points in {n_batches} batches...")

    for i in tqdm(range(0, len(points), batch_size), total=n_batches):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
    print(f"Successfully uploaded {len(points)} points in {n_batches} batches.")

def process_and_upload_chunks(collection_name: str, chunks: list[dict], batch_size: int = 10):
    """Processa os chunks, garante que a coleção exista no Qdrant e faz o upload dos pontos."""
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    collection_exists = client.collection_exists(collection_name=collection_name)
    
    if not collection_exists:
        print(f"Coleção '{collection_name}' não encontrada. Criando nova coleção...")
        client.create_collection(
            collection_name=collection_name,
            # Parâmetro específico para vetores DENSOS
            vectors_config={
                "dense": models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE
                ),
                "sparse": models.VectorParams(
                    size=50000,  
                    distance=models.Distance.COSINE
                )
            },
            # Parâmetro específico para vetores ESPARSOS
            sparse_vectors_config={
                "sparse": models.SparseVectorParams()
            }
        )
        print("Coleção criada com sucesso.")
    else:
        print(f"Coleção '{collection_name}' já existe. Prosseguindo com o upload.")
    
    embedding_models = initialize_embedding_models()
    print("Modelos de embedding inicializados com sucesso.")
    
    points = []
    for chunk in tqdm(chunks, desc="Criando embeddings"):
        point = prepare_point(chunk, embedding_models)
        if point:
            points.append(point)
    
    if points:
        upload_in_batches(client, collection_name, points, batch_size)
        collection_info = client.get_collection(collection_name)
        print(f"Informações da coleção '{collection_name}': {collection_info}")
    else:
        print("Nenhum ponto válido foi gerado para upload.")

def main():
    """
    Função principal para executar o pipeline completo.
    """
    print("=== PIPELINE DE INGESTÃO - PRODUTOS AGRÍCOLAS ===")
    
    print("1. Carregando dados da planilha...")
    df = load_data(DATA_PATH)
    if df is None:
        print("Erro: Não foi possível carregar a planilha. Encerrando.")
        return
    
    # Mostra informações básicas da planilha
    print(f"   - {len(df)} produtos carregados")
    print(f"   - Colunas: {list(df.columns)}")
    
    # Cria chunks dos produtos
    print("2. Criando chunks dos produtos...")
    chunks = create_product_chunks(df)
    print(f"   - {len(chunks)} chunks criados")
    
    # Mostra exemplo de um chunk processado
    print("\n=== EXEMPLO DE CHUNK PROCESSADO ===")
    if chunks:
        example_chunk = chunks[0]
        print(f"Texto: {example_chunk['text'][:200]}...")
        print(f"Metadados: {example_chunk['metadata']}")
    
    # Envia dados para Qdrant
    print("\n3. Enviando dados para Qdrant...")
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "produto_agricolas")
    process_and_upload_chunks(COLLECTION_NAME, chunks)
    
    print("\n=== PIPELINE CONCLUÍDO COM SUCESSO! ===")


# Execução principal
if __name__ == "__main__":
    main()