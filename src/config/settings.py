from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=['.env', '.env.lambda'],
        env_file_encoding='utf-8',
        extra='ignore'
    )

    # AWS Configuration
    aws_region: str = Field(default="us-east-2")
    bedrock_region: str = Field(default="us-east-2")

    # Bedrock Configuration
    llm_model_name: str = Field(
        "us.meta.llama4-scout-17b-instruct-v1:0",
        validation_alias='LLM_MODEL_NAME'
    )

    # WhatsApp Configuration (opcional para desenvolvimento)
    whatsapp_verify_token: Optional[str] = Field(None, validation_alias='WHATSAPP_VERIFY_TOKEN')
    whatsapp_access_token: Optional[str] = Field(None, validation_alias='WHATSAPP_ACCESS_TOKEN')
    whatsapp_phone_number_id: Optional[str] = Field(None, validation_alias='WHATSAPP_PHONE_NUMBER_ID')

    # Qdrant Configuration
    qdrant_url: Optional[str] = Field(None, validation_alias='QDRANT_URL')
    qdrant_api_key: Optional[str] = Field(None, validation_alias='QDRANT_API_KEY')
    qdrant_collection_name: str = Field("chat-edu", validation_alias='QDRANT_COLLECTION_NAME')
    prefetch_limit: int = Field(25, validation_alias='PREFETCH_LIMIT')

    # Embeddings Configuration
    dense_model_name: str = "amazon.titan-embed-text-v2:0"
    bm25_model_name: str = "Qdrant/bm25"

    # RAG Configuration
    retrieval_limit: int = Field(10, validation_alias='RETRIEVAL_LIMIT')

    # Optional Services
    groq_api_key: Optional[str] = Field(None, validation_alias='GROQ_API_KEY')


settings = Settings()