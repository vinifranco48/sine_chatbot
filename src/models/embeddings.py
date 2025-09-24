from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class SparseVector(BaseModel):
    indices: List[int]
    values: List[float]


class QueryEmbeddings(BaseModel):
    dense: List[float]
    sparse_bm25: SparseVector


class Document(BaseModel):
    page_content: str
    metadata: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str = Field(..., description="Pergunta ou consulta para o agente")
    session_id: str = Field(default="default", description="ID da sessão para contexto")

class QueryResponse(BaseModel):
    response: str = Field(..., description="Resposta do agente")
    session_id: str = Field(..., description="ID da sessão")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados adicionais")
