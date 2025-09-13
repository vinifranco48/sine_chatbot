# config/settings.py (VERSÃO SIMPLIFICADA - SEM CORS)
import os
from typing import List, Optional, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, ValidationInfo
import traceback

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=['.env', '../.env', '../../.env'],  # Busca em múltiplos locais
        env_file_encoding='utf-8',
        extra='ignore'
    )

    aws_region: str = Field(default="us-east-1", description="AWS region for all services")
    aws_profile: Optional[str] = Field(default=None, description="AWS profile name (for local development)")
    bedrock_region: str = Field(default="us-east-1", description="AWS Bedrock region")
    bedrock_api: str = Field(..., validation_alias='BEDROCK_API')
    llm_model_name: str = Field(
        "meta.llama4-maverick-17b-instruct-v1:0",
        validation_alias='LLM_MODEL_NAME'
    )
    data_path: str = Field(validation_alias='DATA_PATH')

    qdrant_timeout: float = 60.0

    dense_model_name: str = (
        "amazon.titan-embed-text-v2:0"
    )
    bm25_model_name: str = "Qdrant/bm25"                                                            
    pdf_dir: str = Field("./data/", validation_alias='PDF_DIR')
    prefetch_limit: int = 25
    qdrant_mode: str = Field("url", validation_alias='QDRANT_MODE')
    qdrant_url: Optional[str] = Field(None, validation_alias='QDRANT_URL')
    qdrant_api_key: Optional[str] = Field(None, validation_alias='QDRANT_API_KEY')
    collection_name: str = Field("chat-edu", validation_alias='QDRANT_COLLECTION_NAME')
    prefetch_limit: int = 25


    # --- Configurações de Processamento de Documentos ---
    chunk_size: int = Field(2000, validation_alias='CHUNK_SIZE')
    chunk_overlap: int = Field(200, validation_alias='CHUNK_OVERLAP')

    # --- Configurações do Grafo ---
    retrieval_limit: int = Field(10, validation_alias='RETRIEVAL_LIMIT')

    # --- Cache para embeddings ---
    embedding_cache_dir: str = Field("embedding_cache", validation_alias="EMBEDDING_CACHE_DIR")
    
    # --- Configuração GROQ ---
    groq_api_key: Optional[str] = Field(None, validation_alias='GROQ_API_KEY')

# --- Instanciação das Configurações ---
try:
    # Debug: verificar possíveis localizações do arquivo .env
    possible_env_paths = ['.env', '../.env', '../../.env']
    env_found = False
    
    for env_path in possible_env_paths:
        if os.path.exists(env_path):
            print(f"Arquivo .env encontrado em: {os.path.abspath(env_path)}")
            env_found = True
            
            # Carregar manualmente as variáveis do .env
            from dotenv import load_dotenv
            load_dotenv(env_path)
            
            # Verificar se GROQ_API_KEY está definida
            groq_key = os.getenv('GROQ_API_KEY')
            if groq_key:
                print(f"GROQ_API_KEY encontrada: {groq_key[:10]}...")
            else:
                print("GROQ_API_KEY NÃO encontrada no ambiente.")
            break
    
    if not env_found:
        print("Arquivo .env NÃO encontrado em nenhum dos locais esperados:")
        for path in possible_env_paths:
            print(f"  - {os.path.abspath(path)}")
        print("Certifique-se de que o arquivo .env está no diretório correto.")
    
    settings = Settings()
    print("Instância 'settings' criada com sucesso em config/settings.py (sem CORS).")
except Exception as e:
    print(f"Erro crítico ao INSTANCIAR Settings em config/settings.py:")
    print(f"Erro: {e}")
    traceback.print_exc()
    print("Verifique seu arquivo .env e a LÓGICA dentro de config/settings.py.")
    
    # Tentar carregar manualmente
    print("\n--- Tentando carregar manualmente ---")
    try:
        from dotenv import load_dotenv
        load_dotenv('../.env')  # Tentar carregar do diretório pai
        manual_settings = Settings()
        print("Settings carregado manualmente com sucesso!")
        settings = manual_settings
    except Exception as manual_error:
        print(f"Erro no carregamento manual: {manual_error}")
        exit(1)