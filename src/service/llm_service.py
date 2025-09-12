import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

class LLMService:
    def __init__(self):
        self._init_bedrock()

    def _init_bedrock(self):
        try:
            session_kwargs = {
                "region_name": settings.aws_region
            }
            