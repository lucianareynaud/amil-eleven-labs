#!/bin/bash

# Verificação e download do modelo Mistral se necessário
download_mistral_model() {
    # Diretório para armazenar os modelos
    mkdir -p models

    # Url e nome do arquivo do modelo
    MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    MODEL_FILENAME="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    MODEL_PATH="models/$MODEL_FILENAME"

    # Verificar se o arquivo já existe
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Modelo Mistral não encontrado. Baixando..."
        curl -L "$MODEL_URL" -o "$MODEL_PATH"
        echo "Download concluído: $MODEL_PATH"
    else
        echo "Modelo Mistral já existe em $MODEL_PATH"
    fi
}

# Verificação do arquivo .env
check_env_file() {
    if [ ! -f .env ]; then
        echo "Arquivo .env não encontrado. Criando um template básico..."
        cat > .env << EOL
# ElevenLabs Configuration
ELEVENLABS_API_KEY=your_api_key_here
ELEVENLABS_VOICE_ID=ErXwobaYiN019PkySvjV
ELEVENLABS_MODEL_ID=eleven_multilingual_v1
ELEVENLABS_STABILITY=0.50
ELEVENLABS_SIMILARITY_BOOST=0.75
EOL
        echo "Por favor, edite o arquivo .env e insira sua API key do ElevenLabs"
        exit 1
    fi
}

# Função principal
main() {
    # Verifica parâmetros
    DEV_MODE=false
    
    for arg in "$@"; do
        case $arg in
            --dev)
                DEV_MODE=true
                ;;
            *)
                echo "Uso: $0 [--dev]"
                echo "  --dev: Inicia em modo de desenvolvimento com hot-reload"
                exit 1
                ;;
        esac
    done
    
    # Verifica e baixa modelo se necessário
    download_mistral_model
    
    # Verifica arquivo .env
    check_env_file
    
    # Prepara o docker-compose.yml para o modo selecionado
    if [ "$DEV_MODE" = true ]; then
        echo "Iniciando em modo de desenvolvimento com hot-reload..."
        # Descomenta as linhas para desenvolvimento no docker-compose.yml
        sed -i.bak 's/# - \.\/backend:\/app\/backend/- \.\/backend:\/app\/backend/' docker-compose.yml
        sed -i.bak 's/# command: python -m uvicorn run_ura_service:app --reload --host 0.0.0.0 --port 8000/command: python -m uvicorn run_ura_service:app --reload --host 0.0.0.0 --port 8000/' docker-compose.yml
        docker-compose up --build
        # Restaura o arquivo original após a execução
        mv docker-compose.yml.bak docker-compose.yml
    else
        echo "Iniciando em modo de produção..."
        docker-compose up --build
    fi
}

# Executa a função principal
main "$@" 