import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Carregando modelo spaCy para português
nlp = spacy.load("pt_core_news_sm")

# 2. Leitura e Preparação dos Dados
def ler_mensagens(file_path):
    mensagens = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if ':' in line:
                label, texto = line.strip().split(':', 1)
                mensagens.append({'label': label.strip(), 'message': texto.strip()})
    return mensagens

# 3. Pré-processamento com spaCy
def preprocessar(mensagens):
    mensagens_processadas = []
    for item in mensagens:
        # Aplicando spaCy: minúsculas, remoção stopwords, lematização
        doc = nlp(item['message'].lower())
        
        # Preservando palavras importantes para melhor compraçao
        palavras_importantes = {'não', 'nao', 'nunca', 'muito', 'mais', 'menos', 'melhor', 'pior', 'excelente', 'pessimo'}
        
        tokens = []
        for token in doc:
            if (not token.is_punct and token.is_alpha and
                (not token.is_stop or token.text in palavras_importantes)):
                tokens.append(token.lemma_)
        
        if tokens:
            texto_limpo = ' '.join(tokens)
            mensagens_processadas.append({'label': item['label'], 'message': texto_limpo})
    
    return mensagens_processadas

# 4. Modelagem de IA
def treinar_classificador(mensagens):
    # Separando textos e labels
    textos = [m['message'] for m in mensagens]
    labels = [m['label'] for m in mensagens]
    
    # Vetorização
    vectorizer = TfidfVectorizer(
        max_features=300,      # reduzido para evitar overfitting
        ngram_range=(1, 1),    # Apenas unigrams para simplicidade
        min_df=1,              # mantém todas as palavras
        max_df=0.95            # Remove apenas palavras muito frequentes
    )
    X = vectorizer.fit_transform(textos)
    
    # Divisão treino/teste ajustada para dataset pequeno
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=42, stratify=labels
    )
    
    # Treinamento com Naive Bayes
    modelo = MultinomialNB(alpha=0.5)  # smothing reduzido
    modelo.fit(X_train, y_train)
    
    return modelo, vectorizer, X_test, y_test

# 5. Classificação e Avaliação
def avaliar_modelo(modelo, vectorizer, X_test, y_test):
    # Fazendo predições
    y_pred = modelo.predict(X_test)
    
    # Calculando precisão
    precisao = accuracy_score(y_test, y_pred)
    
    print(f"Precisao do modelo: {precisao:.2%}")
    print("\nRelatorio detalhado:")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusao')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.show()
    
    return y_pred

# Função para classificar novas mensagens
def classificar_nova_mensagem(texto, modelo, vectorizer):
    # Pré-processamento da nova mensagem
    doc = nlp(texto.lower())
    
    # Preservando palavras importantes
    palavras_importantes = {'não', 'nao', 'nunca', 'muito', 'mais', 'menos', 'melhor', 'pior', 'excelente', 'pessimo'}
    
    tokens = []
    for token in doc:
        if (not token.is_punct and token.is_alpha and
            (not token.is_stop or token.text in palavras_importantes)):
            tokens.append(token.lemma_)
    
    if tokens:
        texto_limpo = ' '.join(tokens)
        X_novo = vectorizer.transform([texto_limpo])
        predicao = modelo.predict(X_novo)[0]
        return predicao
    return "Erro no processamento"

# 6. Automação - Execução completa
if __name__ == "__main__":
    print("=== CLASSIFICADOR DE SENTIMENTOS ===")
    
    # Leitura dos dados
    print("1. Carregando mensagens...")
    mensagens = ler_mensagens('mensagens.txt')
    print(f"   {len(mensagens)} mensagens carregadas")
    
    # Pré-processamento
    print("2. Pre-processando textos...")
    mensagens_processadas = preprocessar(mensagens)
    print(f"   {len(mensagens_processadas)} mensagens processadas")
    
    # Treinamento
    print("3. Treinando modelo...")
    modelo, vectorizer, X_test, y_test = treinar_classificador(mensagens_processadas)
    print("   Modelo treinado com sucesso")
    
    # Avaliacão
    print("4. Avaliando performance...")
    y_pred = avaliar_modelo(modelo, vectorizer, X_test, y_test)
    
    # Teste com exemplos
    print("5. Testando com novas mensagens:")
    exemplos = [
        "Este produto e excelente",
        "Pessimo atendimento",
        "Muito satisfeito",
        "Nao recomendo",
        "Isso não vai"
    ]
    
    for exemplo in exemplos:
        resultado = classificar_nova_mensagem(exemplo, modelo, vectorizer)
        print(f"   '{exemplo}' -> {resultado}")
    
    print("\n=== PROCESSO CONCLUIDO ===")