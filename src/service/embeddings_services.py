import boto3
from fastembed.sparse.bm25 import Bm25
from src.models.embeddings import QueryEmbeddings, SparseVector
import os
import json
from typing import Optional


class QueryEmbedder:
    def __init__(self, dense_model_name: str, sparse_model_name: str, aws_region: str = "us-east-2"):
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Usar Amazon Bedrock para embeddings densos (Titan)
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)
        self.dense_model_name = dense_model_name
        
        # Usar FastEmbed para embeddings esparsos (BM25)
        self.sparse_embedding_model = Bm25(sparse_model_name)

    def _get_bedrock_embedding(self, text: str) -> list:
        """Gera embedding denso usando Amazon Titan via Bedrock"""
        try:
            body = json.dumps({
                "inputText": text
            })
            
            response = self.bedrock_client.invoke_model(
                modelId=self.dense_model_name,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['embedding']
            
        except Exception as e:
            raise RuntimeError(f"Erro ao gerar embedding denso com Amazon Titan: {e}")

    def embed_query(self, query: str) -> QueryEmbeddings:
        if not query or not query.strip():
            raise ValueError("Query não pode ser vazia")
            
        try:
            # Embedding denso via Amazon Titan (Bedrock)
            dense_vector = self._get_bedrock_embedding(query)
            
            # Embedding esparso (BM25) via FastEmbed
            sparse_vector = next(self.sparse_embedding_model.query_embed([query]))

            return QueryEmbeddings(
                dense=dense_vector,
                sparse_bm25=SparseVector(**sparse_vector.as_object())
            )
        except Exception as e:
            raise RuntimeError(f"Erro ao gerar embeddings: {e}")