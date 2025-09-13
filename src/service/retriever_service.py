from typing import List
from qdrant_client import QdrantClient
from src.models.embeddings import Document, QueryEmbeddings
from src.config.settings import Settings
from qdrant_client.http.exceptions import UnexpectedResponse
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


class QdrantRetriever:
    def __init__(self, settings: Settings):
        client_params = {'url': settings.qdrant_url, 'api_key': settings.qdrant_api_key}

        if settings.qdrant_api_key:
            client_params['api_key'] = settings.qdrant_api_key

        self.client = QdrantClient(**client_params)
        self.collection_name = settings.qdrant_collection_name
        self.prefetch_limit = settings.prefetch_limit

    def search_documents(
                self, embeddings: QueryEmbeddings, limit: int = 5
                ) -> List[Document]:
            try:
                search_result = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        {
                            "query": embeddings.dense,
                            "using": "dense",
                            "limit": self.prefetch_limit
                        },
                        {
                            "query": embeddings.sparse_bm25.model_dump(),
                            "using": "sparse",
                            "limit": self.prefetch_limit
                        }
                    ]

                )

                return [
                    Document(
                        page_content=point.payload.get("text", ""),
                        metadata = point.payload.get("metadata", {})
                    )
                    for point in search_result.points
                ]
            except UnexpectedResponse as e:
                # Handle Qdrant-specific errors
                logger.error(
                    "Qdrant search failed",
                    extra={"error": str(e), "collection": self.collection_name},
                )
                raise HTTPException(
                    status_code=503, detail="Search service temporarily unavailable"
                )
            except Exception as e:
                # Handle any other errors
                logger.error(
                    "Unexpected error during search",
                    extra={"error": str(e), "collection": self.collection_name},
                )
                raise HTTPException(status_code=500, detail="Internal server error")


    
                         