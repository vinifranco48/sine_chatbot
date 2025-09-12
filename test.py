import boto3
import json
import os
from botocore.exceptions import ClientError

def comprehensive_bedrock_test():
    """Teste abrangente para diagnosticar o problema do Bedrock"""
    
    print("🔍 === DIAGNÓSTICO COMPLETO DO BEDROCK ===\n")
    
    # 1. Testar diferentes regiões
    regions_to_test = [
        'us-east-1', 'us-west-2', 'eu-west-1', 
        'ap-northeast-1', 'ap-southeast-1', 'ca-central-1'
    ]
    
    # 2. Testar diferentes IDs de modelos Titan
    titan_models_to_test = [
        "amazon.titan-embed-text-v2:0",     # Versão que você está tentando
        "amazon.titan-embed-text-v1",       # Versão anterior
        "amazon.titan-embed-g1-text-02",    # Versão G1
        "amazon.titan-tg1-large"            # Alternativa
    ]
    
    working_combinations = []
    
    for region in regions_to_test:
        print(f"\n🌍 === TESTANDO REGIÃO: {region} ===")
        
        try:
            # Teste 1: Cliente de listagem
            bedrock_list = boto3.client('bedrock', region_name=region)
            response = bedrock_list.list_foundation_models()
            
            # Filtrar apenas modelos Titan embedding
            titan_models = [
                model for model in response['modelSummaries'] 
                if 'titan' in model['modelId'].lower() and 
                ('embed' in model['modelId'].lower() or 'text' in model['modelId'].lower())
            ]
            
            print(f"   📋 Modelos Titan disponíveis na região:")
            for model in titan_models:
                status = model.get('modelLifecycle', {}).get('status', 'Unknown')
                print(f"      - {model['modelId']} ({status})")
            
            # Teste 2: Cliente runtime para cada modelo
            bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
            
            for model_id in titan_models_to_test:
                try:
                    print(f"\n   🧪 Testando modelo: {model_id}")
                    
                    # Teste com Titan V2 (formato novo)
                    if "v2" in model_id:
                        body = json.dumps({
                            "inputText": "teste",
                            "dimensions": 256,  # Usar dimensão menor para teste
                            "normalize": True
                        })
                    else:
                        # Teste com formato antigo
                        body = json.dumps({
                            "inputText": "teste"
                        })
                    
                    response = bedrock_runtime.invoke_model(
                        body=body,
                        modelId=model_id,
                        accept="application/json",
                        contentType="application/json"
                    )
                    
                    result = json.loads(response['body'].read())
                    embeddings = result.get('embedding', [])
                    
                    print(f"      ✅ SUCESSO! Dimensões: {len(embeddings)}")
                    working_combinations.append({
                        'region': region,
                        'model_id': model_id,
                        'dimensions': len(embeddings),
                        'format': 'v2' if 'v2' in model_id else 'v1'
                    })
                    
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    print(f"      ❌ {error_code}: {e.response['Error']['Message']}")
                except Exception as e:
                    print(f"      ❌ Erro: {str(e)[:100]}...")
                    
        except Exception as region_error:
            print(f"   ❌ Região {region} inacessível: {region_error}")
    
    # 3. Resumo dos resultados
    print(f"\n🎯 === RESUMO DOS RESULTADOS ===")
    
    if working_combinations:
        print(f"✅ Encontradas {len(working_combinations)} combinações funcionais:")
        for combo in working_combinations:
            print(f"   🌍 Região: {combo['region']}")
            print(f"   🤖 Modelo: {combo['model_id']}")
            print(f"   📏 Dimensões: {combo['dimensions']}")
            print(f"   📝 Formato: {combo['format']}")
            print()
        
        # Recomendar o melhor
        best = working_combinations[0]
        print(f"💡 RECOMENDAÇÃO: Use esta configuração no seu código:")
        print(f"   REGIÃO: '{best['region']}'")
        print(f"   MODELO: '{best['model_id']}'")
        
    else:
        print("❌ NENHUMA combinação funcionou!")
        print("\n💡 Soluções possíveis:")
        print("1. Solicitar acesso aos modelos no console AWS Bedrock")
        print("2. Verificar permissões IAM da sua conta")
        print("3. Aguardar ativação automática (Sep 29, 2025)")
        print("4. Usar modelo local temporariamente")

if __name__ == "__main__":
    comprehensive_bedrock_test()