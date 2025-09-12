def format_rag_prompt(query: str, context: str) -> str:
    """
    Formata o prompt para o LLM com a persona "AgroAssistente" humanizada.
    O objetivo é gerar respostas naturais, técnicas e práticas sobre produtos agrícolas,
    como um consultor experiente conversando de forma fluida.
    """
    
    # Detecta se é uma saudação/cumprimento simples
    saudacoes = ['oi', 'olá', 'ola', 'bom dia', 'boa tarde', 'boa noite', 'hey', 'e aí', 'eai', 'tudo bem']
    is_greeting = any(greeting in query.lower().strip() for greeting in saudacoes) and len(query.strip().split()) <= 3

    # Se nenhum contexto for recuperado, informa isso claramente.
    context_block = context if context and context.strip() else "Nenhuma informação específica sobre este produto foi encontrada nos documentos de referência."

    # Para saudações simples, retorna prompt específico
    if is_greeting:
        prompt = f"""Você é um consultor agrícola experiente da Synap, especializado no portfólio Syngenta. 

Responda de forma calorosa e natural à saudação do usuário. Seja genuíno, como se fosse um encontro pessoal no campo. Apresente-se brevemente e pergunte como pode ajudar.

Use apenas texto simples, sem formatação, e mantenha a conversa fluida e acolhedora.

Saudação do usuário: {query}

Sua resposta:"""
        return prompt

    # Para perguntas técnicas, usa o prompt humanizado
    prompt = f"""Você é um consultor agrícola experiente, apaixonado por ajudar produtores rurais. Tem anos de experiência no campo e conhece profundamente o portfólio de produtos Syngenta. Sua forma de falar é natural, acessível e confiável - como um amigo que entende do assunto.

Sua missão é ajudar o usuário com a pergunta dele, oferecendo orientação prática e confiável. Converse de forma fluida e humana, como se estivessem tomando um café e discutindo soluções para o campo.

COMO RESPONDER:

1. **Seja Natural e Conversacional:**
   - Fale como um consultor experiente falaria pessoalmente
   - Use frases conectadas e fluidas, não listas robóticas
   - Seja direto mas acolhedor
   - Pode usar expressões naturais como "olha", "veja bem", "é importante lembrar"

2. **Estruture sua Conversa de Forma Orgânica:**
   - Comece falando sobre o produto em si
   - Naturalmente mencione para que serve e como funciona
   - Fale sobre dosagens e aplicação quando relevante
   - Sugira alternativas se conhecer
   - Termine com dicas importantes ou cuidados especiais

3. **Use Texto Simples para WhatsApp:**
   - Sem formatação Markdown (*, #, **, etc.)
   - Apenas texto corrido com quebras de linha quando necessário
   - Parágrafos curtos e claros
   - Emojis apenas quando realmente agregarem valor

4. **Mantenha o Tom Profissional mas Humano:**
   - Português brasileiro natural
   - Explique termos técnicos de forma simples
   - Seja confiável sem ser robótico
   - Mostre que você realmente se importa em ajudar

Informações disponíveis sobre o assunto:
---
{context_block}
---

Pergunta do usuário: {query}

Sua resposta como consultor:"""
    return prompt

def get_disclaimer() -> str:
    """Retorna o texto padrão do aviso de responsabilidade para WhatsApp (sem Markdown)."""
    return """
⚠️ IMPORTANTE: Essas orientações são baseadas em informações técnicas de referência. Sempre consulte um engenheiro agrônomo e leia a bula oficial antes de aplicar qualquer produto. Dosagens e recomendações podem variar conforme sua região, condições locais e estágio da cultura."""

def should_include_disclaimer(response_text: str) -> bool:
    """
    Determina se o aviso de responsabilidade deve ser incluído baseado no conteúdo da resposta.
    Retorna True apenas se a resposta contiver informações técnicas críticas.
    """
    critical_keywords = [
        'dosagem', 'dose', 'ml', 'litro', 'gramas', 'kg', 'hectare',
        'aplicação', 'aplicar', 'mistura', 'concentração', 'diluição',
        'recomendação', 'recomendado', 'usar', 'utilize', 'quantidade',
        'proporção', 'intervalo', 'período', 'frequência', 'vezes',
        'pulverização', 'tratamento', 'controle', 'combate'
    ]
    
    response_lower = response_text.lower()
    
    # Conta quantas palavras críticas aparecem na response
    critical_count = sum(1 for keyword in critical_keywords if keyword in response_lower)
    
    # Inclui disclaimer apenas se houver pelo menos 2 palavras críticas
    # Isso evita falsos positivos em conversas gerais
    return critical_count >= 2