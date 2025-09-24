import json
import logging
from typing import Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from src.config.settings import Settings
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._initialize_bedrock_client()
    
    def _initialize_bedrock_client(self) -> None:
        """Initialize the Bedrock client with proper error handling."""
        try:
            self.bedrock_client = boto3.client(
                service_name="bedrock-runtime", 
                region_name=self.settings.bedrock_region,
            )
            logger.info(f"Bedrock client initialized successfully for region: {self.settings.bedrock_region}")
        except NoCredentialsError as e:
            error_msg = "AWS credentials not found. Please configure your credentials."
            logger.error(error_msg)
            raise Exception(error_msg) from e
        except Exception as e:
            error_msg = f"Error initializing Bedrock client: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def generate_response(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_gen_len: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Optional[str]:
        """ Generate a response using the Bedrock LLM model. """
        self._validate_prompt(prompt)
        
        temperature = temperature if temperature is not None else 0.7
        max_gen_len = max_gen_len if max_gen_len is not None else 2048
        top_p = top_p if top_p is not None else 0.9
        
        try:
            body = {
                "prompt": f"Human: {prompt}\n\nAssistant:",
                "max_gen_len": max_gen_len,
                "temperature": temperature,
                "top_p": top_p
            }
            
            logger.debug(f"Invoking model {self.settings.llm_model_name} with prompt length: {len(prompt)}")

            response = self.bedrock_client.invoke_model(
                body=json.dumps(body),
                modelId=self.settings.llm_model_name,
                accept="application/json",
                contentType="application/json"
            )
            
            response_body = json.loads(response.get('body').read())
            generated_text = self._extract_response_text(response_body)
            
            if generated_text:
                logger.info(f"Successfully generated response of length: {len(generated_text)}")
            else:
                logger.warning("No text could be extracted from the response")
                
            return generated_text
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Bedrock API Error [{error_code}]: {error_message}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {str(e)}")
            return None
    
    def _validate_prompt(self, prompt: str) -> None:
        """ Validate the input prompt. """
        if not prompt:
            raise ValueError("Prompt cannot be None")
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty or only whitespace")
    
    def _extract_response_text(self, response_body: dict) -> Optional[str]:
        """ Extract text from various Bedrock response formats. """
        if 'generation' in response_body:
            return response_body['generation']
        elif 'completions' in response_body and response_body['completions']:
            return response_body['completions'][0].get('data', {}).get('text')
        elif 'results' in response_body and response_body['results']:
            return response_body['results'][0].get('outputText')
        elif 'output' in response_body:
            return response_body['output'].get('text')
        else:
            logger.warning(f"Unknown response format. Available keys: {list(response_body.keys())}")
            return None