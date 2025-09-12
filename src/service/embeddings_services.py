from fastembed import TextEmbedding
from fastembed.sparse.bm25 import Bm25
from src.models.embeddings import QueryEmbeddings, SparseVector
import os


class QueryEmbedder:
    def __init__(self, dense_model_name: str, sparse_model_name: str):
            if "TOKENIZER_PARALLELISM" not in os.environ:
                os.environ["TOKENIZER_PARALLELISM"] = "false"

            self.dense_embeddings_models = TextEmbedding(
                 dense_model_name
            
            )
            self.sparse_embeddings_models = Bm25(sparse_model_name)

    
    def embed_query(self, query: str) -> QueryEmbeddings:
         dense_vector =  next(self.dense_embeddings_models.embed([query])).tolist()
         sparse_vector = next(self.sparse_embeddings_models.passage_embed([query]))

         return QueryEmbeddings(
              dense=dense_vector,
              sparse_bm25=SparseVector(**sparse_vector.as_object())
         )

         

