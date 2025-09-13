from pydantic import BaseModel
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