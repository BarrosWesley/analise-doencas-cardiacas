import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# --- 1. Configuração e Carregamento de Dados ---
NÍVEL_LOG_TF = 'ERROR'
pd.options.mode.chained_assignment = None
np.random.seed(100)

CAMINHO_DO_ARQUIVO = "heart.csv"
NOMES_DAS_COLUNAS = [
    'idade', 'sexo', 'tipo_dor_peito', 'pressao_arterial_repouso', 'colesterol',
    'acucar_jejum', 'eletrocardiograma_repouso', 'frequencia_cardiaca_maxima',
    'angina_induzida_exercicio', 'depressao_st', 'inclinacao_st',
    'numero_principais_vasos', 'talassemia', 'alvo'
]
COLUNA_ALVO = 'alvo'

dados = pd.read_csv(CAMINHO_DO_ARQUIVO)
dados.columns = NOMES_DAS_COLUNAS

# --- 2. Transformação dos Dados ---
def codificar_caracteristicas_categoricas(df):
    """Codifica características categóricas para strings legíveis para humanos."""
    mapeamento = {
        'sexo': {0: 'feminino', 1: 'masculino'},
        'tipo_dor_peito': {1: 'angina tipica', 2: 'angina atipica', 3: 'dor nao-anginosa', 4: 'assintomatico'},
        'acucar_jejum': {0: 'menor que 120mg/ml', 1: 'maior que 120mg/ml'},
        'eletrocardiograma_repouso': {0: 'normal', 1: 'anormalidade da onda ST-T', 2: 'hipertrofia ventricular esquerda'},
        'angina_induzida_exercicio': {0: 'nao', 1: 'sim'},
        'inclinacao_st': {1: 'ascendente', 2: 'plano', 3: 'descendente'},
        'talassemia': {1: 'normal', 2: 'defeito fixo', 3: 'defeito reversivel'}
    }
    for coluna, mapa_codificacao in mapeamento.items():
        if coluna in df.columns:
            df[coluna] = df[coluna].map(mapa_codificacao)
    return df

dados = codificar_caracteristicas_categoricas(dados.copy())
dados = pd.get_dummies(dados, drop_first=True)

# --- 3. Escalonamento de Características ---
def normalizar_dataframe(df):
    """Normaliza todas as colunas numéricas no DataFrame para o intervalo [0, 1]."""
    df_normalizado = df.copy()
    colunas_para_remover = []  # Lista para armazenar colunas com variação zero

    for coluna in df_normalizado.columns:
        if pd.api.types.is_numeric_dtype(df_normalizado[coluna]):
            df_normalizado[coluna] = df_normalizado[coluna].fillna(0).astype(np.float64)  # Preencher NaN e converter para float64

            min_valor = df_normalizado[coluna].min()
            max_valor = df_normalizado[coluna].max()

            # Verificação de valores constantes
            if max_valor > min_valor:
                df_normalizado[coluna] = (df_normalizado[coluna] - min_valor) / (max_valor - min_valor)
            elif max_valor == min_valor:
                colunas_para_remover.append(coluna)
            else:
                df_normalizado[coluna] = 0

            # Verificação de infinitos
            if np.isinf(df_normalizado[coluna]).any():
                print(f"Erro: Coluna '{coluna}' contém infinitos após normalização.")

    # Remover colunas com variação zero
    df_normalizado = df_normalizado.drop(columns=colunas_para_remover, errors='ignore')
    return df_normalizado

dados = normalizar_dataframe(dados.copy())

# --- 4. Visualização 1: Heatmap de Correlação ---
plt.figure(figsize=(12, 10))
sns.heatmap(dados.corr(), annot=True, cmap='coolwarm')
plt.title("Matriz de Correlação das Características")
plt.show()

# --- 5. Divisão dos Dados ---
X = dados.drop(COLUNA_ALVO, axis=1, errors='ignore')
y = dados[COLUNA_ALVO]
x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=0)

# --- 6. Treinamento e Avaliação dos Modelos ---
def treinar_e_avaliar_modelo(modelo, nome_modelo, x_treino, y_treino, x_teste, y_teste):
    """Treina e avalia um dado modelo."""
    modelo.fit(x_treino, y_treino)
    y_predito = modelo.predict(x_teste)

    # Garantir que as previsões sejam numéricas (0 ou 1)
    if y_predito.dtype == bool:
        y_predito = y_predito.astype(int)

    acuracia = accuracy_score(y_teste, y_predito)
    print(f"{nome_modelo} Acurácia: {acuracia:.4f}")
    return modelo, y_predito, acuracia

print("\n--- Desempenho dos Modelos Individuais ---")
modelos_treinados = {}
resultados = []
modelos_lista = [
    (LogisticRegression(solver='lbfgs', random_state=0), "Regressão Logística"),
    (KNeighborsClassifier(), "K-Vizinhos Mais Próximos"),
    (SVC(gamma='auto', random_state=0), "Máquina de Vetores de Suporte"),
    (GaussianNB(), "Naive Bayes"),
    (DecisionTreeClassifier(random_state=0), "Árvore de Decisão"),
    (RandomForestClassifier(n_estimators=100, random_state=0), "Floresta Aleatória"),
    (MLPClassifier(hidden_layer_sizes=(100, 100, 10), max_iter=300, random_state=0), "Rede Neural")
]

for modelo, nome in modelos_lista:
    modelo_treinado, predito, acuracia = treinar_e_avaliar_modelo(modelo, nome, x_treino, y_treino, x_teste, y_teste)
    resultados.append((nome, acuracia))
    modelos_treinados[nome] = modelo_treinado

# --- 7. Visualização 2: Gráfico de Barras da Acurácia dos Modelos ---
nomes_modelos = [r[0] for r in resultados]
acuracias_modelos = [r[1] for r in resultados]

plt.figure(figsize=(10, 6))
plt.bar(nomes_modelos, acuracias_modelos, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.xlabel("Modelo")
plt.ylabel("Acurácia")
plt.title("Acurácia dos Modelos Individuais")
plt.tight_layout()
plt.show()

def avaliar_classificacao(y_verdadeiro, y_predito, nome_modelo="Modelo"):
    """Avalia um modelo de classificação usando métricas comuns."""
    print(f"\n--- Métricas de Avaliação para {nome_modelo} ---")
    matriz_confusao = confusion_matrix(y_verdadeiro, y_predito)
    print("Matriz de Confusão:\n", matriz_confusao)
    total = np.sum(matriz_confusao)
    sensibilidade = matriz_confusao[1, 1] / (matriz_confusao[1, 1] + matriz_confusao[1, 0]) if np.sum(matriz_confusao[1,:]) > 0 else 0
    especificidade = matriz_confusao[0, 0] / (matriz_confusao[0, 0] + matriz_confusao[0, 1]) if np.sum(matriz_confusao[0,:]) > 0 else 0
    acuracia = accuracy_score(y_verdadeiro, y_predito)
    precisao = precision_score(y_verdadeiro, y_predito) if np.sum(matriz_confusao[:, 1]) > 0 else 0
    revocacao = recall_score(y_verdadeiro, y_predito) if np.sum(matriz_confusao[1,:]) > 0 else 0

    print(f'Especificidade: {especificidade:.4f}')
    print(f'Sensibilidade (Revocação): {sensibilidade:.4f}')
    print(f'Acurácia: {acuracia:.4f}')
    print(f'Precisão: {precisao:.4f}')
    return matriz_confusao

# --- 8. Modelo de Rede Neural (usando scikit-learn) ---
modelo_neural_treinado = modelos_treinados.get("Rede Neural")
predito_neural = modelo_neural_treinado.predict(x_teste)
acuracia_neural = accuracy_score(y_teste, predito_neural)
print(f"\nAcurácia da Rede Neural: {acuracia_neural:.4f}")
matriz_confusao_nn = confusion_matrix(y_teste, predito_neural)
print("Matriz de Confusão (RN):\n", matriz_confusao_nn)

print("\n--- Avaliação da Rede Neural ---")
avaliar_classificacao(y_teste, predito_neural, "Rede Neural")

# --- 9. Classificador de Votação ---
print("\n--- Classificador de Votação ---")

# Garantir que TODAS as previsões sejam numéricas antes de somar
predicoes = [
    modelos_treinados.get("Regressão Logística").predict(x_teste).astype(int),
    modelos_treinados.get("Máquina de Vetores de Suporte").predict(x_teste).astype(int),
    modelos_treinados.get("Naive Bayes").predict(x_teste).astype(int),
    modelos_treinados.get("Floresta Aleatória").predict(x_teste).astype(int),
    modelos_treinados.get("Árvore de Decisão").predict(x_teste).astype(int),
    modelos_treinados.get("K-Vizinhos Mais Próximos").predict(x_teste).astype(int),
    predito_neural.astype(int)  # Já calculado
]

votos = np.sum(np.array(predicoes), axis=0)  # Soma ao longo das linhas (modelos)

predito_voto_maioria = (votos >= 4).astype(int)
print("\nEstatísticas para o classificador de votação (maioria simples):")
avaliar_classificacao(y_teste, predito_voto_maioria, "Classificador de Votação (Maioria)")

predito_voto_dois = (votos >= 2).astype(int)
print("\nEstatísticas para o classificador de votação (pelo menos 2 votos positivos):")
avaliar_classificacao(y_teste, predito_voto_dois, "Classificador de Votação (>= 2 Votos)")

# --- 10. Visualização 3: Distribuição da Variável Alvo ---
plt.figure(figsize=(6, 4))
sns.countplot(x=COLUNA_ALVO, data=dados)
plt.title("Distribuição da Variável Alvo")
plt.xlabel("Alvo")
plt.ylabel("Contagem")
plt.show()