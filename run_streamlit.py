#!/usr/bin/env python3
"""
Script para executar a interface Streamlit do AgroAssistente
"""
import subprocess
import sys
import os

def check_streamlit_installed():
    """Verifica se o Streamlit está instalado"""
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} encontrado")
        return True
    except ImportError:
        print("❌ Streamlit não encontrado")
        return False

def install_streamlit_requirements():
    """Instala os requisitos do Streamlit"""
    print("📦 Instalando dependências do Streamlit...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_streamlit.txt"
        ])
        print("✅ Dependências instaladas com sucesso")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False

def run_streamlit():
    """Executa a aplicação Streamlit"""
    print("🚀 Iniciando AgroAssistente Streamlit...")
    print("🌐 A aplicação será aberta no seu navegador")
    print("🔗 URL: http://localhost:8501")
    print("\n" + "="*50)
    
    try:
        subprocess.run([
            "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 AgroAssistente encerrado pelo usuário")
    except FileNotFoundError:
        print("❌ Comando 'streamlit' não encontrado")
        print("💡 Tente instalar com: pip install streamlit")

def main():
    """Função principal"""
    print("🌾 AgroAssistente - Interface Streamlit")
    print("="*40)
    
    # Verificar se está no diretório correto
    if not os.path.exists("streamlit_app.py"):
        print("❌ Arquivo streamlit_app.py não encontrado")
        print("💡 Execute este script no diretório raiz do projeto")
        return
    
    # Verificar e instalar Streamlit se necessário
    if not check_streamlit_installed():
        print("📦 Instalando Streamlit...")
        if not install_streamlit_requirements():
            return
    
    # Executar aplicação
    run_streamlit()

if __name__ == "__main__":
    main()