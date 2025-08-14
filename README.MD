Este projeto realiza analise de texto (positivas ou negativas) usando **NLP com spaCy** e **machine learning com scikit-learn**, 
foi desenvolvido para processar mensagens, treinar um modelo de classificação e realizar previsões em novos textos.

## Estrutura esperada

Desafio de desenvolvimento IA/
├── venv/
├── mensagens.txt - Arquivo com as mensagens no formato "label: mensagem"
├── Analise.py - Script principal do projeto
└── requirements.txt - Dependências do projeto

## Instalação:

# 1. Crie e ative um ambiente virtual
python -m venv venv
venv\Scripts\activate  # No Windows

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Baixe o modelo spaCy para português
python -m spacy download pt_core_news_sm

## Após as instalações apenas executar o comando:
python Analise.py

# --------------------------------------------------
## Observações importantes
# --------------------------------------------------
Se estiver rodando o projeto no Google Colab, várias bibliotecas já estão pré-instaladas.

Caso esteja em uma rede com proxy corporativo ou firewall, será obrigatório rodar em um local virtual seguindo o 
passo da instalação se não pode pular o passo 1 e ir direto para 2 nas instalações 