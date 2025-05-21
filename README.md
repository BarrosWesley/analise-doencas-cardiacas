# Análise de Doenças Cardíacas com Aprendizado de Máquina

Este projeto realiza uma análise preditiva de doenças cardíacas utilizando diversas técnicas de aprendizado de máquina. O objetivo é construir modelos capazes de prever a presença ou ausência de doenças cardíacas com base em um conjunto de características clínicas.

## Visão Geral

O projeto segue um fluxo de trabalho típico de aprendizado de máquina, incluindo:

1.  **Carregamento e Pré-processamento de Dados:** Carregamento do conjunto de dados, tratamento de valores categóricos e normalização das características.
2.  **Análise Exploratória de Dados (EDA):** Visualização da distribuição da variável alvo e da correlação entre as características.
3.  **Treinamento e Avaliação de Modelos:** Treinamento de vários modelos de classificação (Regressão Logística, KNN, SVM, Naive Bayes, Árvore de Decisão, Floresta Aleatória e Rede Neural) e avaliação de seu desempenho.
4.  **Classificador de Votação (Ensemble):** Implementação de um classificador de votação para combinar as previsões dos modelos individuais.

## Tecnologias Utilizadas

* **Python:** Linguagem de programação principal.
* **Pandas:** Para manipulação e análise de dados.
* **NumPy:** Para computação numérica.
* **Scikit-learn:** Para algoritmos de aprendizado de máquina.
* **Matplotlib:** Para visualização de dados.
* **Seaborn:** Para visualização de dados estatísticos (especialmente o heatmap).

## Conjunto de Dados

O conjunto de dados utilizado é o "Heart Disease Dataset", que contém informações sobre pacientes e suas características clínicas relacionadas a doenças cardíacas. O arquivo `heart.csv` deve estar presente no mesmo diretório do script Python.

## Estrutura do Código

O código Python (`main.py`) é organizado nas seguintes seções:

1.  **Configuração e Carregamento de Dados:** Carrega o conjunto de dados e define os nomes das colunas.
2.  **Transformação dos Dados:** Codifica características categóricas e normaliza as características numéricas.
3.  **Escalonamento de Características:** Normaliza as colunas numéricas para o intervalo \[0, 1].
4.  **Visualização 1: Heatmap de Correlação:** Gera um heatmap para visualizar a correlação entre as características.
5.  **Divisão dos Dados:** Divide o conjunto de dados em conjuntos de treino e teste.
6.  **Treinamento e Avaliação dos Modelos:** Treina e avalia vários modelos de classificação.
7.  **Visualização 2: Gráfico de Barras da Acurácia dos Modelos:** Cria um gráfico de barras para comparar a acurácia dos modelos.
8.  **Modelo de Rede Neural (usando scikit-learn):** Treina e avalia um modelo de rede neural.
9.  **Classificador de Votação:** Combina as previsões dos modelos individuais usando votação.
10. **Visualização 3: Distribuição da Variável Alvo:** Exibe a distribuição da variável alvo.

## Como Executar

1.  **Pré-requisitos:**
    * Python 3.x
    * Bibliotecas Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn) - Você pode instalá-las usando `pip`:
        ```bash
        pip install pandas numpy scikit-learn matplotlib seaborn
        ```
2.  **Execução:**
    * Certifique-se de que o arquivo `heart.csv` esteja no mesmo diretório do script `main.py`.
    * Execute o script Python:
        ```bash
        python main.py
        ```

## Resultados e Análise

O script executa o treinamento e a avaliação de diversos modelos de classificação, além de um classificador de votação, para prever a presença de doenças cardíacas.

### Desempenho dos Modelos Individuais

A acurácia de cada modelo individual no conjunto de teste foi a seguinte:

* **Regressão Logística:** 0.8197
* **K-Vizinhos Mais Próximos:** 0.7869
* **Máquina de Vetores de Suporte (SVM):** 0.8361
* **Naive Bayes:** 0.7705
* **Árvore de Decisão:** 0.8033
* **Floresta Aleatória:** 0.8525
* **Rede Neural (MLPClassifier):** 0.8197

**Observação:** A Rede Neural (MLPClassifier) pode apresentar um aviso de `ConvergenceWarning`, indicando que o otimizador estocástico atingiu o número máximo de iterações e a otimização ainda não convergiu totalmente. Isso pode ser ajustado aumentando `max_iter` ou otimizando outros hiperparâmetros.

### Avaliação Detalhada da Rede Neural

Além da acurácia, avaliamos a Rede Neural com métricas mais detalhadas:

* **Matriz de Confusão:**
    ```
    [[22  5]
     [ 6 28]]
    ```
    * **Verdadeiros Positivos (TP):** 28 (pacientes com doença cardíaca corretamente previstos)
    * **Verdadeiros Negativos (TN):** 22 (pacientes sem doença cardíaca corretamente previstos)
    * **Falsos Positivos (FP):** 5 (pacientes sem doença cardíaca previstos incorretamente)
    * **Falsos Negativos (FN):** 6 (pacientes com doença cardíaca previstos incorretamente)

* **Especificidade:** 0.8148 (Proporção de verdadeiros negativos; capacidade do modelo de identificar corretamente quem *não* tem a doença.)
* **Sensibilidade (Revocação):** 0.8235 (Proporção de verdadeiros positivos; capacidade do modelo de identificar corretamente quem *tem* a doença.)
* **Acurácia:** 0.8197 (Proporção de previsões corretas no total.)
* **Precisão:** 0.8485 (Proporção de previsões positivas que foram realmente corretas.)

### Classificador de Votação (Ensemble)

O classificador de votação combina as previsões de todos os modelos individuais, incluindo a Rede Neural. Avaliamos duas estratégias de votação:

#### 1. Maioria Simples (4 ou mais votos positivos)

* **Matriz de Confusão:**
    ```
    [[24  3]
     [ 4 30]]
    ```
* **Especificidade:** 0.8889
* **Sensibilidade (Revocação):** 0.8824
* **Acurácia:** 0.8852
* **Precisão:** 0.9091

**Análise:** Esta estratégia de votação demonstrou um desempenho superior aos modelos individuais em todas as métricas. A acurácia aumentou para **0.8852**, e tanto a especificidade quanto a sensibilidade foram significativamente altas, indicando um bom equilíbrio na identificação de casos positivos e negativos. A alta precisão (0.9091) é particularmente importante em contextos médicos, pois significa que a maioria das previsões de doença cardíaca são de fato corretas.

#### 2. Pelo Menos 2 Votos Positivos (de 7 modelos)

* **Matriz de Confusão:**
    ```
    [[20  7]
     [ 2 32]]
    ```
* **Especificidade:** 0.7407
* **Sensibilidade (Revocação):** 0.9412
* **Acurácia:** 0.8525
* **Precisão:** 0.8205

**Análise:** Ao reduzir o limiar de votação para apenas 2 votos positivos, a **sensibilidade (recall)** aumentou consideravelmente para **0.9412**. Isso significa que o modelo se tornou muito melhor em identificar pacientes que *realmente* têm a doença cardíaca (menos falsos negativos). No entanto, houve uma pequena queda na especificidade e precisão, indicando um aumento nos falsos positivos. Esta estratégia pode ser preferível em cenários onde é crucial não perder nenhum caso de doença, mesmo que isso signifique um número maior de falsos alarmes.

### Conclusão dos Resultados

A Floresta Aleatória foi o modelo individual com melhor desempenho. No entanto, o **Classificador de Votação (maioria simples)** superou todos os modelos individuais, atingindo a maior acurácia e um excelente equilíbrio entre sensibilidade e especificidade. Isso reforça o poder das abordagens de *ensemble* em problemas de classificação complexos. A escolha da estratégia de votação ideal dependerá do custo relativo de falsos positivos e falsos negativos no contexto da aplicação real.

## Contribuição

Contribuições são bem-vindas! Se você tiver sugestões de melhorias ou correções de bugs, sinta-se à vontade para abrir uma issue ou enviar um pull request.

## Autor

Wesley Barros

## Licença

MIT License
