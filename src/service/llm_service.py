import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from src.config.settings import Settings
from dotenv import load_dotenv
load_dotenv()   
settings = Settings()

class LLMService:
    def __init__(self):
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime", 
            region_name='us-east-2',
        )
        body = {
            "prompt": "Human: <pergunta>\n\nAssistant:",
            "max_gen_len": 2048,
            "temperature": 0.7,
            "top_p": 0.9
            }

        self.bedrock_client.invoke_model(
            body=json.dumps(body),
            modelId=settings.llm_model_name,
            accept="application/json",
            contentType="application/json"
        )
    
    def generate_response(self, prompt: str) -> str | None:
        try:
            chat_completion = self. 
        
            