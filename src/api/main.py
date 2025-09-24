import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from mangum import Mangum
from typing import Dict, Any
from src.core.agent import create_compiled_graph, Agent
from src.service.embeddings_services import QueryEmbedder
from src.service.retriever_service import QdrantRetriever
from src.service.llm_service import LLMService
from src.config.settings import Settings
from src.models.embeddings import QueryRequest, QueryResponse

app = FastAPI(
    title="Agente Inteligente API",
    description="API serverless para agente inteligente com LangGraph",
    version="1.0.0"
)
logger = logging.getLogger(__name__)
settings = Settings()

query_embedder = None
retriever = None
llm_service = None
agent = None
compiled_graph = None

def get_services():
    """ Initializes and returns the services """
    global query_embedder, retriever, llm_service, agent, compiled_graph
    if compiled_graph is None:
        try:
            logger.info("Initializing services...")
            dense_model = settings.dense_model_name
            sparse_model = settings.bm25_model_name

            query_embedder = QueryEmbedder(
                dense_model_name=dense_model,
                sparse_model_name=sparse_model,
                aws_region=settings.aws_region
            )
            retriever = QdrantRetriever(settings=settings)
            llm_service = LLMService(settings=settings)

            # Create the compiled graph with correct parameter names
            compiled_graph = create_compiled_graph(
                query_embedder=query_embedder,
                qdrant_retriever=retriever,
                llm_service=llm_service
            )
            logger.info("Services initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            raise e
    return compiled_graph

@app.get("/")
async def root():
    """Endpoint de health check"""
    return {"message": "Agente Inteligente API está funcionando!", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Endpoint detalhado de health check"""
    return {
        "status": "healthy",
        "service": "agente-inteligente",
        "version": "1.0.0"
    }
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Endpoint to process user queries and return responses.
    """
    try:
        logger.info(f"Processing query: {request.query}")

        graph = get_services()

        result = await graph.ainvoke({
            "query": request.query,
            "session_id": request.session_id
        })

        return QueryResponse(
            response=result.get("response", "Consult failed"),
            session_id=request.session_id,
            metadata=result.get("metadata", {})
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

handler = Mangum(app)