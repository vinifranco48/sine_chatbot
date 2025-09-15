"""
AWS Lambda Function - SINE Chatbot WhatsApp Integration
========================================================

Este é o ponto de entrada principal para o chatbot SINE no AWS Lambda.
Integra WhatsApp Business API com o sistema RAG existente.

Fluxo completo:
1. API Gateway recebe webhook do WhatsApp
2. Lambda processa evento (verificação ou mensagem)
3. Se mensagem: extrai texto → chama agent.py → responde via WhatsApp API
4. Retorna resposta formatada para API Gateway

Autor: Claude Code - Tutorial AWS Lambda + WhatsApp
"""

import json
import os
import logging
import traceback
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime

# Configuração de logging para Lambda
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    🚀 ENTRY POINT PRINCIPAL DO AWS LAMBDA

    Esta função é chamada automaticamente quando:
    - API Gateway recebe uma requisição HTTP
    - Event contém dados da requisição (método, path, body, headers, etc.)
    - Context contém informações do Lambda (request_id, timeout, etc.)

    Args:
        event: Dados da requisição HTTP do API Gateway
        context: Contexto de execução do Lambda

    Returns:
        Dict formatado para API Gateway com statusCode, headers e body
    """
    # Log do request ID para debugging
    request_id = context.aws_request_id if hasattr(context, 'aws_request_id') else 'local'
    logger.info(f"🔄 Iniciando processamento - RequestID: {request_id}")
    logger.info(f"📨 Evento recebido - Método: {event.get('httpMethod', 'UNKNOWN')} | Path: {event.get('path', 'UNKNOWN')}")

    try:
        # Extrair informações básicas da requisição
        http_method = event.get('httpMethod', '').upper()
        path = event.get('path', '').rstrip('/')  # Remove barra final se houver

        # 🔀 ROTEAMENTO PRINCIPAL
        # Cada endpoint tem uma responsabilidade específica

        if http_method == 'GET' and path == '/webhook':
            # 📋 VERIFICAÇÃO DO WEBHOOK (obrigatório Meta)
            # Meta envia GET para verificar se nosso endpoint é válido
            logger.info("🔍 Processando verificação de webhook")
            return verify_whatsapp_webhook(event)

        elif http_method == 'POST' and path == '/webhook':
            # 💬 PROCESSAMENTO DE MENSAGEM (core do chatbot)
            # Meta envia POST quando usuário manda mensagem
            logger.info("💬 Processando mensagem do WhatsApp")
            return process_whatsapp_message(event)

        elif http_method == 'GET' and path == '/health':
            # 🏥 HEALTH CHECK (para monitoramento)
            # Verifica se Lambda está funcionando
            logger.info("🏥 Processando health check")
            return health_check()

        else:
            # ❌ ENDPOINT NÃO ENCONTRADO
            logger.warning(f"❌ Endpoint não encontrado: {http_method} {path}")
            return create_error_response(404, f'Endpoint {http_method} {path} não encontrado')

    except Exception as e:
        # 🚨 TRATAMENTO DE ERRO GLOBAL
        # Captura qualquer erro não previsto
        error_details = traceback.format_exc()
        logger.error(f"🚨 Erro crítico não tratado: {str(e)}")
        logger.error(f"📋 Stack trace: {error_details}")

        return create_error_response(500, 'Erro interno do servidor')


def verify_whatsapp_webhook(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    📋 VERIFICAÇÃO DO WEBHOOK WHATSAPP

    Quando você configura o webhook no Meta for Developers, eles enviam
    uma requisição GET com parâmetros especiais para verificar se o
    endpoint está funcionando e se você é o dono legítimo.

    Processo:
    1. Meta envia: GET /webhook?hub.mode=subscribe&hub.challenge=123&hub.verify_token=seu_token
    2. Verificamos se o token confere com o nosso
    3. Se confere: retornamos o challenge
    4. Se não confere: retornamos erro 403

    Args:
        event: Evento do API Gateway contendo queryStringParameters

    Returns:
        Response com challenge (sucesso) ou erro 403
    """
    logger.info("🔍 Iniciando verificação do webhook WhatsApp")

    try:
        # Extrair parâmetros da query string
        # API Gateway coloca eles em event['queryStringParameters']
        query_params = event.get('queryStringParameters') or {}

        # Parâmetros enviados pela Meta
        mode = query_params.get('hub.mode')           # Deve ser 'subscribe'
        verify_token = query_params.get('hub.verify_token')  # Nosso token secreto
        challenge = query_params.get('hub.challenge')        # String que devemos retornar

        # Token que configuramos nas variáveis de ambiente
        expected_token = os.environ.get('WHATSAPP_VERIFY_TOKEN')

        logger.info(f"📋 Parâmetros recebidos - Mode: {mode} | Token: {verify_token[:10] if verify_token else 'None'}...")
        logger.info(f"🔑 Token esperado: {expected_token[:10] if expected_token else 'None'}...")

        # Validação dos parâmetros
        if not expected_token:
            logger.error("❌ WHATSAPP_VERIFY_TOKEN não configurado nas variáveis de ambiente")
            return create_error_response(500, 'Token de verificação não configurado')

        if mode == 'subscribe' and verify_token == expected_token:
            # ✅ VERIFICAÇÃO BEM-SUCEDIDA
            logger.info("✅ Webhook verificado com sucesso! Retornando challenge.")
            # IMPORTANTE: retornar apenas o challenge, não JSON
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'text/plain'},
                'body': challenge
            }
        else:
            # ❌ TOKEN INVÁLIDO
            logger.warning(f"❌ Verificação falhou - Mode: {mode} | Token válido: {verify_token == expected_token}")
            return create_error_response(403, 'Token de verificação inválido')

    except Exception as e:
        logger.error(f"🚨 Erro na verificação do webhook: {str(e)}")
        return create_error_response(500, 'Erro na verificação do webhook')


def process_whatsapp_message(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    💬 PROCESSAMENTO DE MENSAGENS DO WHATSAPP

    Esta é a função mais importante! Aqui acontece toda a mágica:
    1. Recebe mensagem do usuário via webhook
    2. Extrai texto e dados do remetente
    3. Processa com seu sistema RAG (agent.py)
    4. Envia resposta de volta via WhatsApp API

    Formato do webhook da Meta:
    {
      "entry": [{
        "changes": [{
          "value": {
            "messages": [{
              "from": "5511999999999",
              "text": {"body": "Como me cadastrar no SINE?"},
              "timestamp": "1234567890"
            }]
          }
        }]
      }]
    }

    Args:
        event: Evento contendo body com dados do WhatsApp

    Returns:
        Response indicando sucesso ou erro no processamento
    """
    logger.info("💬 Iniciando processamento de mensagem WhatsApp")

    try:
        # 📨 PARSE DO BODY JSON
        # API Gateway coloca o body como string, precisamos fazer parse
        raw_body = event.get('body', '{}')
        logger.info(f"📨 Body bruto recebido (primeiros 500 chars): {raw_body[:500]}...")

        try:
            whatsapp_data = json.loads(raw_body)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Erro ao fazer parse do JSON: {str(e)}")
            return create_error_response(400, 'JSON inválido no body da requisição')

        # 🔍 EXTRAIR DADOS DA MENSAGEM
        # WhatsApp manda estrutura complexa, precisamos extrair o essencial
        message_info = extract_whatsapp_message(whatsapp_data)

        if not message_info:
            # Pode ser notificação de status, webhook de teste, etc.
            logger.info("ℹ️ Nenhuma mensagem de usuário encontrada (pode ser status/notificação)")
            return create_success_response({'status': 'no_user_message', 'processed': True})

        logger.info(f"📱 Mensagem extraída - De: {message_info['from']} | Texto: '{message_info['text'][:100]}...'")

        # 🤖 PROCESSAR COM SEU CHATBOT
        # Aqui é onde seu sistema RAG entra em ação!
        logger.info("🤖 Enviando para o sistema RAG...")
        bot_response = process_with_your_agent(message_info['text'])

        # 📤 ENVIAR RESPOSTA VIA WHATSAPP API
        logger.info("📤 Enviando resposta via WhatsApp API...")
        send_success = send_whatsapp_message(message_info['from'], bot_response)

        if send_success:
            logger.info("✅ Mensagem processada e enviada com sucesso!")
            return create_success_response({
                'status': 'success',
                'message_processed': True,
                'response_sent': True,
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            logger.error("❌ Falha ao enviar mensagem via WhatsApp API")
            return create_error_response(500, 'Falha ao enviar resposta')

    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"🚨 Erro ao processar mensagem: {str(e)}")
        logger.error(f"📋 Stack trace: {error_details}")
        return create_error_response(500, 'Erro ao processar mensagem')


def extract_whatsapp_message(whatsapp_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    🔍 EXTRAÇÃO DE DADOS DA MENSAGEM WHATSAPP

    O WhatsApp manda uma estrutura JSON complexa. Esta função navega
    pela estrutura e extrai apenas o que precisamos: número do remetente
    e texto da mensagem.

    Estrutura típica:
    {
      "entry": [
        {
          "changes": [
            {
              "value": {
                "messages": [
                  {
                    "from": "5511999999999",
                    "text": {"body": "Mensagem do usuário"},
                    "type": "text",
                    "timestamp": "1234567890"
                  }
                ]
              }
            }
          ]
        }
      ]
    }

    Args:
        whatsapp_data: JSON completo recebido do webhook

    Returns:
        Dict com 'from' e 'text' ou None se não encontrar mensagem
    """
    logger.info("🔍 Extraindo dados da mensagem WhatsApp")

    try:
        # Navegar pela estrutura aninhada do WhatsApp
        # Usar .get() para evitar KeyError se estrutura for diferente
        entries = whatsapp_data.get('entry', [])

        if not entries:
            logger.info("ℹ️ Nenhuma entrada encontrada no webhook")
            return None

        for entry in entries:
            changes = entry.get('changes', [])

            for change in changes:
                value = change.get('value', {})
                messages = value.get('messages', [])

                for message in messages:
                    # Verificar se é mensagem de texto de usuário
                    message_type = message.get('type')

                    if message_type == 'text':
                        # Extrair dados essenciais
                        from_number = message.get('from')
                        text_content = message.get('text', {}).get('body')
                        timestamp = message.get('timestamp')

                        if from_number and text_content:
                            logger.info(f"📱 Mensagem encontrada - De: {from_number} | Texto: '{text_content[:50]}...'")

                            return {
                                'from': from_number,
                                'text': text_content.strip(),
                                'timestamp': timestamp,
                                'type': message_type
                            }
                    else:
                        logger.info(f"ℹ️ Tipo de mensagem ignorado: {message_type}")

        logger.info("ℹ️ Nenhuma mensagem de texto encontrada")
        return None

    except Exception as e:
        logger.error(f"❌ Erro ao extrair mensagem: {str(e)}")
        return None


def process_with_your_agent(user_message: str) -> str:
    """
    🤖 INTEGRAÇÃO COM SEU SISTEMA RAG

    Esta função conecta a mensagem do WhatsApp com seu código existente
    em src/core/agent.py. Aqui acontece toda a inteligência:

    1. Carrega configurações (settings.py)
    2. Inicializa serviços (embeddings, retriever, LLM)
    3. Cria o grafo LangGraph
    4. Executa o workflow RAG completo
    5. Retorna resposta gerada

    Args:
        user_message: Texto da mensagem enviada pelo usuário

    Returns:
        Resposta gerada pelo sistema RAG
    """
    logger.info(f"🤖 Processando com agent RAG: '{user_message[:100]}...'")

    try:
        # 📥 IMPORTAR SEU CÓDIGO EXISTENTE
        # Aqui conectamos com todo o sistema que você já construiu
        from src.core.agent import create_compiled_graph
        from src.service.embeddings_services import QueryEmbedder
        from src.service.retriever_service import QdrantRetriever
        from src.service.llm_service import LLMService
        from src.config.settings import Settings

        logger.info("📚 Módulos importados com sucesso")

        # 🔧 CARREGAR CONFIGURAÇÕES
        # Usa as configurações que você já definiu, agora com suporte Lambda
        settings = Settings()
        logger.info(f"⚙️ Configurações carregadas - Qdrant: {settings.qdrant_url[:30] if settings.qdrant_url else 'None'}...")

        # 🚀 INICIALIZAR SERVIÇOS
        # Os mesmos serviços que você usa, mas agora no Lambda
        logger.info("🚀 Inicializando serviços...")

        query_embedder = QueryEmbedder(settings)
        logger.info("✅ QueryEmbedder inicializado")

        qdrant_retriever = QdrantRetriever(settings)
        logger.info("✅ QdrantRetriever inicializado")

        llm_service = LLMService(settings)
        logger.info("✅ LLMService inicializado")

        # 🔗 CRIAR GRAFO LANGGRAPH
        # Seu workflow: embed_query → retrieve_documents → generate_response
        compiled_graph = create_compiled_graph(
            query_embedder=query_embedder,
            qdrant_retriever=qdrant_retriever,
            llm_service=llm_service
        )
        logger.info("🔗 Grafo LangGraph compilado")

        # 🎯 EXECUTAR WORKFLOW RAG
        # Aqui roda todo o pipeline que você criou
        logger.info("🎯 Executando workflow RAG...")
        result = compiled_graph.invoke({
            "query": user_message,
            "filters": None  # Você pode adicionar filtros se necessário
        })

        # 📤 EXTRAIR RESPOSTA
        response_text = result.get('response')

        if response_text:
            logger.info(f"✅ Resposta gerada com sucesso (tamanho: {len(response_text)} chars)")
            logger.info(f"📝 Prévia da resposta: '{response_text[:150]}...'")
            return response_text
        else:
            # Fallback se algo der errado
            logger.warning("⚠️ Resposta vazia do sistema RAG")
            return "Desculpe, não consegui processar sua pergunta no momento. Tente novamente."

    except ImportError as e:
        logger.error(f"❌ Erro ao importar módulos: {str(e)}")
        return "Erro interno: módulo não encontrado."

    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"🚨 Erro no processamento RAG: {str(e)}")
        logger.error(f"📋 Stack trace: {error_details}")
        return "Desculpe, ocorreu um erro ao processar sua pergunta. Nossa equipe foi notificada."


def send_whatsapp_message(to_number: str, message_text: str) -> bool:
    """
    📤 ENVIO DE MENSAGEM VIA WHATSAPP BUSINESS API

    Usa a Meta Graph API para enviar mensagem de volta ao usuário.
    Requer access token e phone number ID configurados.

    Endpoint: POST https://graph.facebook.com/v18.0/{phone-number-id}/messages

    Payload exemplo:
    {
      "messaging_product": "whatsapp",
      "to": "5511999999999",
      "type": "text",
      "text": {"body": "Resposta do chatbot"}
    }

    Args:
        to_number: Número do destinatário (formato internacional)
        message_text: Texto da mensagem a enviar

    Returns:
        True se enviado com sucesso, False caso contrário
    """
    logger.info(f"📤 Enviando mensagem para {to_number}: '{message_text[:100]}...'")

    try:
        # 🔑 CREDENCIAIS DA META API
        access_token = os.environ.get('WHATSAPP_ACCESS_TOKEN')
        phone_number_id = os.environ.get('WHATSAPP_PHONE_NUMBER_ID')

        if not access_token or not phone_number_id:
            logger.error("❌ Credenciais WhatsApp não configuradas")
            return False

        # 🌐 ENDPOINT DA META GRAPH API
        url = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"

        # 📋 HEADERS DA REQUISIÇÃO
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }

        # 📦 PAYLOAD DA MENSAGEM
        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "text",
            "text": {
                "body": message_text
            }
        }

        logger.info(f"🌐 Enviando para: {url}")
        logger.info(f"📦 Payload: {json.dumps(payload, indent=2)}")

        # 🚀 FAZER REQUISIÇÃO HTTP
        response = requests.post(
            url=url,
            headers=headers,
            json=payload,
            timeout=30  # Timeout de 30 segundos
        )

        # 📊 VERIFICAR RESPOSTA
        logger.info(f"📊 Status da resposta: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            message_id = response_data.get('messages', [{}])[0].get('id')
            logger.info(f"✅ Mensagem enviada com sucesso! ID: {message_id}")
            return True
        else:
            logger.error(f"❌ Erro na API WhatsApp: {response.status_code}")
            logger.error(f"📋 Resposta: {response.text}")
            return False

    except requests.exceptions.Timeout:
        logger.error("⏰ Timeout ao enviar mensagem WhatsApp")
        return False

    except requests.exceptions.RequestException as e:
        logger.error(f"🌐 Erro de rede ao enviar mensagem: {str(e)}")
        return False

    except Exception as e:
        logger.error(f"🚨 Erro inesperado ao enviar mensagem: {str(e)}")
        return False


def health_check() -> Dict[str, Any]:
    """
    🏥 HEALTH CHECK ENDPOINT

    Endpoint simples para verificar se o Lambda está funcionando.
    Útil para monitoramento, load balancers, etc.

    Verifica:
    - Lambda está executando
    - Variáveis de ambiente estão configuradas
    - Conexão com serviços (opcional)

    Returns:
        Status da aplicação com informações básicas
    """
    logger.info("🏥 Executando health check")

    try:
        # Verificar variáveis essenciais
        required_vars = [
            'WHATSAPP_VERIFY_TOKEN',
            'WHATSAPP_ACCESS_TOKEN',
            'WHATSAPP_PHONE_NUMBER_ID',
            'QDRANT_URL',
            'LLM_MODEL_NAME'
        ]

        missing_vars = [var for var in required_vars if not os.environ.get(var)]

        health_status = {
            'status': 'healthy' if not missing_vars else 'unhealthy',
            'service': 'SINE Chatbot Lambda',
            'version': '1.0.0',
            'timestamp': datetime.utcnow().isoformat(),
            'environment': os.environ.get('ENVIRONMENT', 'unknown'),
            'lambda_function': True,
            'config_status': {
                'required_vars_present': len(required_vars) - len(missing_vars),
                'total_required_vars': len(required_vars),
                'missing_vars': missing_vars
            }
        }

        if not missing_vars:
            logger.info("✅ Health check: sistema saudável")
        else:
            logger.warning(f"⚠️ Health check: variáveis faltando: {missing_vars}")

        return create_success_response(health_status)

    except Exception as e:
        logger.error(f"❌ Erro no health check: {str(e)}")
        return create_error_response(500, 'Health check falhou')


def create_success_response(body: Any, status_code: int = 200) -> Dict[str, Any]:
    """
    ✅ CRIAR RESPOSTA DE SUCESSO

    Formata resposta para API Gateway seguindo o padrão esperado.
    Inclui headers CORS para permitir testes no browser.

    Args:
        body: Dados a retornar (dict, list, string, etc.)
        status_code: Código HTTP (padrão 200)

    Returns:
        Dict formatado para API Gateway
    """
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
        },
        'body': json.dumps(body, ensure_ascii=False) if isinstance(body, (dict, list)) else str(body)
    }


def create_error_response(status_code: int, message: str) -> Dict[str, Any]:
    """
    ❌ CRIAR RESPOSTA DE ERRO

    Formata erro para API Gateway com estrutura consistente.

    Args:
        status_code: Código HTTP de erro
        message: Mensagem de erro para o usuário

    Returns:
        Dict formatado para API Gateway
    """
    error_body = {
        'error': True,
        'message': message,
        'statusCode': status_code,
        'timestamp': datetime.utcnow().isoformat()
    }

    return create_success_response(error_body, status_code)


# 🧪 CÓDIGO PARA TESTES LOCAIS
if __name__ == '__main__':
    """
    🧪 TESTE LOCAL DA FUNÇÃO LAMBDA

    Para testar localmente, execute: python lambda_function.py
    """

    # Simular evento de verificação de webhook
    test_webhook_event = {
        'httpMethod': 'GET',
        'path': '/webhook',
        'queryStringParameters': {
            'hub.mode': 'subscribe',
            'hub.challenge': 'test_challenge_123',
            'hub.verify_token': 'sine_chatbot_webhook_verify_2024'
        }
    }

    # Simular evento de mensagem
    test_message_event = {
        'httpMethod': 'POST',
        'path': '/webhook',
        'body': json.dumps({
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "5511999999999",
                            "text": {"body": "Como me cadastrar no SINE?"},
                            "type": "text",
                            "timestamp": "1234567890"
                        }]
                    }
                }]
            }]
        })
    }

    # Contexto mock
    class MockContext:
        aws_request_id = 'test-request-id-123'

    # Executar testes
    print("🧪 Testando verificação de webhook...")
    result1 = lambda_handler(test_webhook_event, MockContext())
    print(f"Resultado: {json.dumps(result1, indent=2)}")

    print("\n🧪 Testando processamento de mensagem...")
    result2 = lambda_handler(test_message_event, MockContext())
    print(f"Resultado: {json.dumps(result2, indent=2)}")
