import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import VectorParams, Distance

load_dotenv()

os.getenv("QDRANT_COLLECTION_NAME")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collections = client.get_collections().collections
collections_names = [collection.name for collection in collections]
if COLLECTION_NAME in collections_names:
    print(f"Collection '{COLLECTION_NAME}' já existe. Removendo")
    client.delete_collection(COLLECTION_NAME)


client.create_collection(
    collections_name=COLLECTION_NAME,
    vectors_config={
        'dense': VectorParams(
            size=1024,
            distance=Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            )
        )
    },
    sparse_vectors_config={
        'sparse': models.SparseVectorsParams(
            modifier=models.Modifier.IDF
        ),
    },
    )


print(f'Create Collection {COLLECTION_NAME}')

collection_info = client.get_collection(COLLECTION_NAME)
print(f"Status: {collection_info.status}")
print(f"Vectors config: {list(collection_info.config.params.vectors.keys())}")
print(
    f"Sparses: {list(collection_info.config.params.sparse_vectors.keys() if collection_info.config.params.sparse_vectors else [])}"
)
print(f"Pointers: {collection_info.points_count}")
