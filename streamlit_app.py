import streamlit as st
import sys
import os
import time
from datetime import datetime

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.agent import create_compiled_graph, Agent
from src.service.embeddings_services import QueryEmbedder
from src.service.retriever_service import QdrantRetriever
from src.service.llm_service import LLMService
from src.config.settings import settings

# Configuração da página
st.set_page_config(
    page_title="🌾 AgroAssistente - Chatbot Agrícola",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E7D32;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #E8F5E8;
        border-left: 4px solid #4CAF50;
    }
    .metrics-container {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_services():
    """Inicializa os serviços do chatbot (cached para performance)"""
    try:
        with st.spinner("Inicializando serviços do AgroAssistente..."):
            query_embedder = QueryEmbedder(
                dense_model_name=settings.bedrock_embedding_model, 
                sparse_model_name=settings.bm25_model_name,
                aws_region=settings.aws_region
            )
            
            qdrant_retriever = QdrantRetriever(settings=settings)
            llm_service = LLMService(settings=settings)
            
            compiled_graph = create_compiled_graph(
                query_embedder=query_embedder,
                qdrant_retriever=qdrant_retriever,
                llm_service=llm_service
            )
            
            return compiled_graph, query_embedder, qdrant_retriever, llm_service
    except Exception as e:
        st.error(f"Erro ao inicializar serviços: {e}")
        return None, None, None, None

def process_query(compiled_graph, query: str):
    """Processa uma query usando o grafo compilado"""
    initial_state = {
        'query': query,
        'filters': None,
        'query_embedding': None,
        'retrieved_docs': [],
        'context': "",
        'response': None,
        'error': None
    }
    
    start_time = time.time()
    result = compiled_graph.invoke(initial_state)
    end_time = time.time()
    
    return result, end_time - start_time

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 AgroAssistente - Chatbot Agrícola</h1>', unsafe_allow_html=True)
    st.markdown("**Seu consultor agrícola especializado em produtos Syngenta**")
    
    # Sidebar com informações
    with st.sidebar:
        st.header("ℹ️ Informações do Sistema")
        
        # Status dos serviços
        st.subheader("🔧 Status dos Serviços")
        compiled_graph, query_embedder, qdrant_retriever, llm_service = initialize_services()
        
        if compiled_graph:
            st.success("✅ Serviços inicializados")
            st.info(f"🤖 LLM: {settings.llm_model_name}")
            st.info(f"🔍 Embeddings: Amazon Titan + BM25")
            st.info(f"🗄️ Qdrant: {settings.qdrant_collection_name}")
            st.info(f"🌍 Região: {settings.aws_region}")
        else:
            st.error("❌ Erro na inicialização")
            return
        
        st.divider()
        
        # Configurações
        st.subheader("⚙️ Configurações")
        show_debug = st.checkbox("Mostrar informações de debug", value=False)
        show_context = st.checkbox("Mostrar contexto recuperado", value=False)
        
        st.divider()
        
        # Exemplos de perguntas
        st.subheader("💡 Exemplos de Perguntas")
        example_questions = [
            "Como plantar milho?",
            "Qual herbicida usar para soja?",
            "Como controlar pragas no tomate?",
            "Quando aplicar fertilizante no café?",
            "Produtos para controle de fungos",
            "Dosagem recomendada para algodão"
        ]
        
        for question in example_questions:
            if st.button(f"📝 {question}", key=f"example_{question}"):
                st.session_state.example_query = question

    # Área principal
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Input da pergunta
        query = st.text_input(
            "🤔 Faça sua pergunta agrícola:",
            placeholder="Ex: Como plantar milho? Qual herbicida usar para soja?",
            value=st.session_state.get('example_query', ''),
            key="main_query"
        )
        
        # Limpar exemplo após usar
        if 'example_query' in st.session_state:
            del st.session_state.example_query
    
    with col2:
        search_button = st.button("🔍 Consultar", type="primary", use_container_width=True)
    
    # Processar query
    if search_button and query.strip():
        with st.spinner("🤖 Processando sua pergunta..."):
            try:
                result, processing_time = process_query(compiled_graph, query.strip())
                
                # Exibir resultado
                if result.get('response') and not result.get('error'):
                    # Resposta principal
                    st.markdown('<div class="chat-message user-message">', unsafe_allow_html=True)
                    st.markdown(f"**👤 Você perguntou:** {query}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="chat-message assistant-message">', unsafe_allow_html=True)
                    st.markdown(f"**🌾 AgroAssistente:** {result['response']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Métricas
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("⏱️ Tempo", f"{processing_time:.2f}s")
                    with col2:
                        st.metric("📄 Documentos", len(result.get('retrieved_docs', [])))
                    with col3:
                        st.metric("🔤 Caracteres", len(result['response']))
                    with col4:
                        st.metric("📝 Palavras", len(result['response'].split()))
                    
                    # Debug info
                    if show_debug:
                        st.subheader("🔍 Informações de Debug")
                        with st.expander("Ver detalhes técnicos"):
                            st.json({
                                "query": result.get('query'),
                                "num_docs_retrieved": len(result.get('retrieved_docs', [])),
                                "context_length": len(result.get('context', '')),
                                "processing_time": f"{processing_time:.2f}s",
                                "timestamp": datetime.now().isoformat()
                            })
                    
                    # Contexto recuperado
                    if show_context and result.get('retrieved_docs'):
                        st.subheader("📚 Documentos Consultados")
                        for i, doc in enumerate(result['retrieved_docs'][:3], 1):
                            with st.expander(f"Documento {i} - {doc.get('metadata', {}).get('product_name', 'N/A')}"):
                                st.text(doc.get('text', '')[:500] + "..." if len(doc.get('text', '')) > 500 else doc.get('text', ''))
                                if doc.get('metadata'):
                                    st.json(doc['metadata'])
                
                else:
                    # Erro
                    error_msg = result.get('error', {}).get('message', 'Erro desconhecido')
                    st.markdown('<div class="error-message">', unsafe_allow_html=True)
                    st.markdown(f"**❌ Erro:** {error_msg}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if show_debug:
                        st.subheader("🔍 Detalhes do Erro")
                        st.json(result.get('error', {}))
                        
            except Exception as e:
                st.error(f"❌ Erro inesperado: {e}")
                if show_debug:
                    st.exception(e)
    
    elif search_button and not query.strip():
        st.warning("⚠️ Por favor, digite uma pergunta antes de consultar.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🌾 AgroAssistente - Powered by Amazon Bedrock, Qdrant & FastEmbed<br>
        Especializado em produtos e soluções Syngenta para agricultura
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()