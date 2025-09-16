import json
import os
import uuid
from aiohttp import ClientError
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Dict, Set
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
EMBED_MODEL = settings.dense_model_name
MAX_TOKENS = 750

# S3 Configuration
S3_BUCKET_NAME = "raw-data-sinergia"
S3_PREFIX = "refined/products/"  # Path where product JSONs are stored

def load_data_from_s3(bucket_name: str, prefix: str, max_files: int = None) -> List[Dict]:
    """Load JSON data from S3."""
    try:
        s3_client = boto3.client('s3', region_name='us-east-2')
        
        # List objects in bucket
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            print(f"No files found in bucket {bucket_name} with prefix {prefix}")
            return []
        
        objects = response['Contents']
        
        # Filter only JSON files
        json_files = [obj for obj in objects if obj['Key'].endswith('.json')]
        
        if max_files:
            json_files = json_files[:max_files]
        
        print(f"Found {len(json_files)} JSON files in S3")
        
        products_data = []
        
        for obj in tqdm(json_files, desc="Loading files from S3"):
            try:
                # Download JSON file
                response = s3_client.get_object(Bucket=bucket_name, Key=obj['Key'])
                content = response['Body'].read().decode('utf-8')
                product_data = json.loads(content)
                
                # Add S3 key as unique identifier
                product_data['s3_key'] = obj['Key']
                products_data.append(product_data)
                
            except Exception as e:
                print(f"Error processing file {obj['Key']}: {e}")
                continue
        
        print(f"Data loaded successfully: {len(products_data)} products")
        return products_data
        
    except Exception as e:
        print(f"Error loading data from S3: {e}")
        return []

def create_product_chunks_from_json(products_data: List[Dict]) -> List[Dict]:
    """Create product chunks from S3 JSON data."""
    chunks = []
    
    for index, product in enumerate(products_data):
        product_text = create_product_text_from_json(product)
        metadata = create_product_metadata_from_json(product, index)
        chunks.append({
            "text": product_text, 
            "metadata": metadata,
            "s3_key": product.get('s3_key')  # Maintain S3 reference
        })
    
    return chunks

def create_product_text_from_json(product: Dict) -> str:
    """Create text representation of product from JSON."""
    text_parts = []
    
    # Map common fields that may be in JSON
    field_mapping = {
        'nome': 'Produto',
        'produto': 'Produto', 
        'name': 'Produto',
        'tipo': 'Tipo',
        'type': 'Tipo',
        'fabricante': 'Fabricante',
        'manufacturer': 'Fabricante',
        'empresa': 'Fabricante',
        'alvo': 'Alvo',
        'target': 'Alvo',
        'aplicacao': 'Aplicação',
        'application': 'Aplicação',
        'dose': 'Dose recomendada',
        'dosage': 'Dose recomendada',
        'caracteristicas': 'Características e Benefícios',
        'benefits': 'Características e Benefícios',
        'descricao': 'Descrição',
        'description': 'Descrição'
    }
    
    # Process all product fields
    for key, value in product.items():
        if key == 's3_key':  # Skip S3 key
            continue
            
        # Use mapping if available, otherwise use original key
        label = field_mapping.get(key.lower(), key.title())
        
        if value and str(value).strip():
            text_parts.append(f"{label}: {value}")
    
    return "\n".join(text_parts)

def create_product_metadata_from_json(product: Dict, index: int) -> Dict:
    """Create metadata for product from JSON."""
    metadata = {
        "chunk_id": index,
        "s3_key": product.get('s3_key', ''),
    }
    
    # Map specific fields to metadata
    field_mapping = {
        'nome': 'product_name',
        'produto': 'product_name',
        'name': 'product_name',
        'tipo': 'product_type',
        'type': 'product_type',
        'fabricante': 'manufacturer',
        'manufacturer': 'manufacturer',
        'empresa': 'manufacturer',
        'alvo': 'target',
        'target': 'target',
        'aplicacao': 'application',
        'application': 'application',
        'dose': 'dosage',
        'dosage': 'dosage',
    }
    
    for key, value in product.items():
        if key == 's3_key':
            continue
            
        metadata_key = field_mapping.get(key.lower())
        if metadata_key and value and str(value).strip():
            metadata[metadata_key] = str(value).strip()
    
    # Remove empty values
    metadata = {k: v for k, v in metadata.items() if v is not None and v != ''}
    
    return metadata

def get_existing_s3_keys(client: QdrantClient, collection_name: str) -> Set[str]:
    """Get S3 keys of products already inserted in collection."""
    existing_keys = set()
    
    try:
        # Scroll through all points to get S3 keys
        offset = None
        limit = 100
        
        while True:
            response = client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=True
            )
            
            points = response[0]
            next_offset = response[1]
            
            for point in points:
                s3_key = point.payload.get('s3_key')
                if s3_key:
                    existing_keys.add(s3_key)
            
            if next_offset is None:
                break
            offset = next_offset
        
        print(f"Found {len(existing_keys)} existing S3 keys in collection")
        
    except Exception as e:
        print(f"Error getting existing keys (collection may not exist): {e}")
        # If collection doesn't exist, return empty set
        
    return existing_keys

def filter_new_chunks(chunks: List[Dict], existing_keys: Set[str]) -> List[Dict]:
    """Filter chunks that haven't been processed yet."""
    new_chunks = []
    
    for chunk in chunks:
        s3_key = chunk.get('s3_key')
        if s3_key not in existing_keys:
            new_chunks.append(chunk)
    
    print(f"Filtered chunks: {len(new_chunks)} new out of {len(chunks)} total")
    return new_chunks

def initialize_embedding_models():
    """Initialize embedding models needed for hybrid search."""
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
    """Create embeddings for text using specified models."""
    try:
        # Request body for Titan Text Embeddings V2
        body = json.dumps({
            "inputText": chunk_text,
            "dimensions": 1024,
            "normalize": True
        })
        
        # Invoke model on AWS
        response = bedrock_client.invoke_model(
            body=body,
            modelId=EMBED_MODEL,
            accept="application/json",
            contentType="application/json"
        )
        
        # Extract embedding vector from response
        response_body = json.loads(response.get("body").read())
        dense_embeddings = response_body.get("embedding")
        
        # Create BM25 embeddings
        bm25_generator = bm25_model.passage_embed([chunk_text])
        bm25_embeddings = list(bm25_generator)[0]
        
        return {
            'dense': dense_embeddings,
            'sparse': bm25_embeddings
        }
        
    except Exception as e:
        print(f"Error creating embeddings: {type(e).__name__}: {e}")
        print(f"Chunk text: {chunk_text[:100]}...")
        raise e

def prepare_point(chunk, embedding_models):
    """Prepare a point to be inserted into vector database."""
    bedrock_client, bm25_model = embedding_models
    
    text = chunk.get("text", "")

    # Check if chunk text is empty
    if not text or not text.strip():
        print(f"Skipping empty chunk. Metadata: {chunk.get('metadata')}")
        return None
    
    embeddings = create_embeddings(text, bedrock_client, bm25_model)
    
    # Validation to skip chunk if embedding creation fails
    if not embeddings or not embeddings.get("dense"):
        print(f"Skipping chunk due to embedding error: {chunk['metadata'].get('product_name')}")
        return None

    # Convert to Qdrant SparseVector
    sparse_vector_qdrant = SparseVector(
        indices=embeddings["sparse"].indices.tolist(),
        values=embeddings["sparse"].values.tolist()
    )
    
    # Unpack metadata
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
    print(f"Upload completed successfully: {len(points)} points in {n_batches} batches.")

def process_and_upload_chunks(collection_name: str, chunks: list[dict], batch_size: int = 10):
    """Process chunks and upload to Qdrant."""
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    collection_exists = client.collection_exists(collection_name=collection_name)
    
    if not collection_exists:
        print(f"Collection '{collection_name}' not found. Creating new collection...")
        client.create_collection(
            collection_name=collection_name,
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
            sparse_vectors_config={
                "sparse": models.SparseVectorParams()
            }
        )
        print("Collection created successfully.")
    else:
        print(f"Collection '{collection_name}' already exists. Checking for duplicates...")
        
        # Get already processed S3 keys
        existing_keys = get_existing_s3_keys(client, collection_name)
        
        # Filter only new chunks
        chunks = filter_new_chunks(chunks, existing_keys)
        
        if not chunks:
            print("All chunks have been processed. Nothing to do.")
            return
    
    embedding_models = initialize_embedding_models()
    print("Embedding models initialized successfully.")
    
    points = []
    for chunk in tqdm(chunks, desc="Creating embeddings"):
        point = prepare_point(chunk, embedding_models)
        if point:
            points.append(point)
    
    if points:
        upload_in_batches(client, collection_name, points, batch_size)
        collection_info = client.get_collection(collection_name)
        print(f"Collection '{collection_name}' info: {collection_info}")
    else:
        print("No valid points were generated for upload.")

def main(max_files_first_batch: int = 1000, max_files_total: int = None):
    """
    Main function to execute the complete pipeline.

    Args:
        max_files_first_batch: Maximum number of files to process in first batch
        max_files_total: Maximum total number of files (None for all)
    """
    print("=== S3 INGESTION PIPELINE - AGRICULTURAL PRODUCTS ===")
    
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "produtos_agricolas")
    
    # Check if data already exists in collection
    try:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        
        if client.collection_exists(collection_name):
            collection_info = client.get_collection(collection_name)
            points_count = collection_info.points_count
            print(f"Existing collection found with {points_count} points")
            
            if points_count == 0:
                print("Empty collection - processing first batch")
                max_files = max_files_first_batch
            else:
                print("Collection with data - processing remaining files")
                max_files = max_files_total
        else:
            print("New collection - processing first batch")
            max_files = max_files_first_batch
            
    except Exception as e:
        print(f"Error checking collection: {e}")
        max_files = max_files_first_batch
    
    print(f"1. Loading data from S3 (maximum: {max_files or 'all'} files)...")
    products_data = load_data_from_s3(S3_BUCKET_NAME, S3_PREFIX, max_files)
    
    if not products_data:
        print("Error: No data was loaded from S3. Terminating.")
        return
    
    print(f"   - {len(products_data)} products loaded")
    
    # Create product chunks
    print("2. Creating product chunks...")
    chunks = create_product_chunks_from_json(products_data)
    print(f"   - {len(chunks)} chunks created")
    
    # Show example of processed chunk
    print("\n=== PROCESSED CHUNK EXAMPLE ===")
    if chunks:
        example_chunk = chunks[0]
        print(f"Texto: {example_chunk['text'][:200]}...")
        print(f"Metadados: {example_chunk['metadata']}")
        print(f"S3 Key: {example_chunk['s3_key']}")
    
    # Send data to Qdrant
    print("\n3. Sending data to Qdrant...")
    process_and_upload_chunks(collection_name, chunks)
    
    print("\n=== PIPELINE COMPLETED SUCCESSFULLY! ===")


# Main execution
if __name__ == "__main__":
    # First execution: process only 1000 files
    # main(max_files_first_batch=1000, max_files_total=None)

    # Subsequent executions: process all files, but skip already processed ones
    main(max_files_first_batch=1000, max_files_total=None)