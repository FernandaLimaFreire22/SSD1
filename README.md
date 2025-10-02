---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    name: python
    version: 3.10
  nbformat: 4
  nbformat_minor: 5
---

::: {#ivj_m26-Gav_ .cell .markdown id="ivj_m26-Gav_"}
# MVP -- Ciência de Dados: Classificação da Condição de Motores {#mvp--ciência-de-dados-classificação-da-condição-de-motores}

**Discente:** Fernanda Lima\
**Data:** 17/09/2025

------------------------------------------------------------------------

## 1. Definição do Problema {#1-definição-do-problema}

O objetivo deste projeto é desenvolver um Mínimo Produto Viável (MVP)
para um problema de classificação, aplicando técnicas de machine
learning em um contexto de manutenção preditiva.

A tarefa consiste em prever a condição de motores (0 = ruim, 1 = bom)
com base em variáveis medidas por sensores, como pressão do óleo,
pressão do combustível, pressão do fluido de arrefecimento, temperaturas
e rotações por minuto (RPM).

Esse tipo de solução pode auxiliar empresas na tomada de decisões de
manutenção, reduzindo falhas inesperadas e otimizando custos
operacionais.

Hipótese

As variáveis coletadas pelos sensores dos motores contêm informações
suficientes para prever corretamente se o motor se encontra em boa
condição (1) ou em má condição (0).
:::

::: {#2_yScZcSGawG .cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="2_yScZcSGawG" outputId="f2fb0492-95e9-4d6f-99cc-0d45edd5b362"}
``` python
# Importação de bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
print("Bibliotecas importadas com sucesso!")
```

::: {.output .stream .stdout}
    Bibliotecas importadas com sucesso!
:::
:::

::: {#aS3kiN-9GawJ .cell .markdown id="aS3kiN-9GawJ"}
## 2. Preparação dos Dados {#2-preparação-dos-dados}
:::

::: {#41DI-hmmGawK .cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="41DI-hmmGawK" outputId="4303a709-ed56-4f05-f5fb-a3fb365ec5b5"}
``` python
# Carregar os dados presentes no git hub e retirados do kaggle
url = 'https://raw.githubusercontent.com/FernandaLimaFreire22/SSD1/main/engine_data2.csv'
df = pd.read_csv(url, sep=';')

# Converter colunas com separador de milhar '.'
cols_to_convert = ['Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
for col in cols_to_convert:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    df[col] = pd.to_numeric(df[col])

print(df.head())
print("Formato do dataset:", df.shape)
```

::: {.output .stream .stdout}
       Engine rpm  Lub oil pressure  Fuel pressure  Coolant pressure  \
    0         700        2493591821     1179092738        3178980794   
    1         876        2941605932     1619386556        2464503704   
    2         520        2961745579     6553146911        1064346764   
    3         473        3707834743     1951017166        3727455362   
    4         619        5672918584     1573887141        2052251454   

       lub oil temp  Coolant temp  Engine Condition  
    0    8414416293     816321865                 1  
    1    7764093415     824457245                 0  
    2    7775226574    7964577667                 1  
    3    7412990715    7177462869                 1  
    4    7839698883    8700022538                 0  
    Formato do dataset: (19535, 7)
:::
:::

::: {#f1e30UCcGawM .cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":408}" id="f1e30UCcGawM" outputId="993e3593-c462-4f31-d5cf-fed6536891f6"}
``` python
# Estatísticas descritivas
display(df.describe().T)

# Distribuição da variável alvo
#Exibe estatísticas descritivas das colunas numéricas e a distribuição absoluta e percentual da variável alvo
#Ou seja, é uma primeira análise exploratória para conhecer os dados e sua distribuição.

print(df['Engine Condition'].value_counts())
print(df['Engine Condition'].value_counts(normalize=True) * 100)
```

::: {.output .display_data}
``` json
{"summary":"{\n  \"name\": \"print(df['Engine Condition']\",\n  \"rows\": 7,\n  \"fields\": [\n    {\n      \"column\": \"count\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.0,\n        \"min\": 19535.0,\n        \"max\": 19535.0,\n        \"num_unique_values\": 1,\n        \"samples\": [\n          19535.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"mean\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 3010829505.9406686,\n        \"min\": 0.6305093422062964,\n        \"max\": 7102537772.961249,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          791.2392628615306\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"std\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1049928724.0383787,\n        \"min\": 0.48267922872651176,\n        \"max\": 2583937276.3528075,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          267.6111931048415\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"min\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2608619.8850903534,\n        \"min\": 0.0,\n        \"max\": 7322759.0,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          61.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"25%\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 3137500254.86462,\n        \"min\": 0.0,\n        \"max\": 7527920222.5,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          593.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"50%\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 3303272726.337131,\n        \"min\": 1.0,\n        \"max\": 7731694344.0,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          746.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"75%\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 3497311250.2987595,\n        \"min\": 1.0,\n        \"max\": 8234316764.0,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          934.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"max\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 4342665274.14216,\n        \"min\": 1.0,\n        \"max\": 9999406461.0,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          2239.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe"}
```
:::

::: {.output .stream .stdout}
    Engine Condition
    1    12317
    0     7218
    Name: count, dtype: int64
    Engine Condition
    1    63.050934
    0    36.949066
    Name: proportion, dtype: float64
:::
:::

::: {#HH6Or38IYSBb .cell .markdown id="HH6Or38IYSBb"}
## 3. Análise de Dados {#3-análise-de-dados}

Esses gráficos permitem visualizar como cada variável numérica do motor
está distribuída. Observamos que algumas variáveis, como Engine rpm,
apresentam uma distribuição mais concentrada, enquanto outras, como
pressões e temperaturas, têm distribuições multimodais, sugerindo que
existem diferentes condições de operação ou registros anômalos. Essa
análise é importante porque ajuda a identificar padrões, possíveis
outliers e diferenças de comportamento entre regimes de funcionamento do
motor, o que pode impactar na detecção de falhas ou na previsão da
condição do motor.

A matriz de correlação mostra que, no conjunto de dados, a maior parte
das variáveis não apresenta relação linear significativa entre si. A
única correlação mais perceptível é entre a rotação do motor (Engine
rpm) e a condição do motor (Engine Condition), que aparece como uma
correlação negativa fraca (-0,27). Isso sugere que, conforme a rotação
aumenta, a condição do motor pode tender a se deteriorar levemente,
enquanto as demais variáveis analisadas se comportam de forma
praticamente independente.
:::

::: {#gN1K172KGawN .cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" id="gN1K172KGawN" outputId="70181357-5c05-44cd-afbe-e747cd6f6f97"}
``` python
# Visualizações - Cria histogramas com curva de densidade para cada variável numérica (exceto a variável alvo) para visualizar sua distribuição
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
num_cols.remove('Engine Condition')

plt.figure(figsize=(14,8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2,3,i)
    sns.histplot(df[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# Matriz de correlação
# Gera e plota a matriz de correlação entre as variáveis numéricas para identificar relações lineares
#As cores vermelho → correlação positiva forte.
#As cores azul → correlação negativa.
#O tom mais próximo do branco → correlação fraca ou nula.

corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Matriz de Correlação")
plt.show()
```

::: {.output .display_data}
![](a658d3f6547e5dc65118e4c44ddd1bacab678b66.png)
:::

::: {.output .display_data}
![](5d1655e5201b7ca751a08d5a1d070e03a3303b55.png)
:::
:::

::: {#ViLN2qoEGawN .cell .markdown id="ViLN2qoEGawN"}
## 4. Modelagem e Treinamento {#4-modelagem-e-treinamento}

O dataset foi dividido em dois subconjuntos: 80% dos dados (15.628
registros) foram reservados para treinar o modelo e 20% (3.907
registros) para testá-lo. Essa separação garante que o modelo seja
avaliado em dados que ele nunca viu, permitindo verificar sua capacidade
de generalização. Além disso, o uso de estratificação assegura que a
distribuição das classes da variável alvo seja mantida tanto no treino
quanto no teste.
:::

::: {#plr5bv3QGawO .cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="plr5bv3QGawO" outputId="74980331-85b8-4611-9920-821a3578dc61"}
``` python
# Separa a variável alvo das preditoras e divide os dados em treino (80%) e teste (20%) de forma estratificada
X = df.drop('Engine Condition', axis=1)
y = df['Engine Condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Treino:", X_train.shape, "Teste:", X_test.shape)
```

::: {.output .stream .stdout}
    Treino: (15628, 6) Teste: (3907, 6)
:::
:::

::: {#M2dEkPf4uI5K .cell .markdown id="M2dEkPf4uI5K"}
O modelo baseline alcançou 63% de acurácia simplesmente por prever
sempre a classe mais frequente (classe 1). No entanto, ele falhou
totalmente em identificar a classe 0, resultando em precisão, recall e
f1-score iguais a 0 para essa categoria. Isso mostra que, apesar da
acurácia parecer razoável, o modelo não é útil para distinguir as
classes. Assim, o baseline serve apenas como referência inicial para
compararmos modelos mais sofisticados, que precisam superar esse
desempenho trivial.
:::

::: {#iR_-lmnNGawQ .cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="iR_-lmnNGawQ" outputId="9be8fd5c-faac-49ce-8264-cb6f847e8e3c"}
``` python
# Baseline
# Cria um modelo baseline que sempre prevê a classe mais frequente e avalia sua acurácia como referência inicial
baseline = DummyClassifier(strategy='most_frequent', random_state=42)
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)
print("Baseline Acurácia:", accuracy_score(y_test, y_pred_base))
print(classification_report(y_test, y_pred_base))
```

::: {.output .stream .stdout}
    Baseline Acurácia: 0.6304069618633222
                  precision    recall  f1-score   support

               0       0.00      0.00      0.00      1444
               1       0.63      1.00      0.77      2463

        accuracy                           0.63      3907
       macro avg       0.32      0.50      0.39      3907
    weighted avg       0.40      0.63      0.49      3907
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    /usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    /usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
:::
:::

::: {#8YLfZMgky0U5 .cell .markdown id="8YLfZMgky0U5"}
Foi treinado um modelo de **Regressão Logística**, que é um algoritmo de
classificação linear. O pipeline aplicou uma padronização das variáveis
antes do treino e, em seguida, o modelo foi avaliado no conjunto de
teste. Ele alcançou 64,6% de acurácia, superando o baseline de 63%, o
que indica que conseguiu extrair alguma informação útil dos dados. No
entanto, como o ganho foi pequeno, isso sugere a necessidade de testar
algoritmos mais robustos (como Random Forest ou XGBoost) e aplicar
técnicas para lidar com o desbalanceamento de classes, a fim de melhorar
a capacidade preditiva do modelo.
:::

::: {#6HHc7_QdGawR .cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="6HHc7_QdGawR" outputId="10418ff0-4355-4a69-d459-209826ca07b6"}
``` python
# Regressão Logística - prever se o motor está em boa ou má condição
pipe_log = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=42, max_iter=1000))
])
pipe_log.fit(X_train, y_train)
y_pred_log = pipe_log.predict(X_test)
print("Acurácia Regressão Logística:", accuracy_score(y_test, y_pred_log))
```

::: {.output .stream .stdout}
    Acurácia Regressão Logística: 0.64601996416688
:::
:::

::: {#BpKZRLRh1Ph4 .cell .markdown id="BpKZRLRh1Ph4"}
O modelo de Random Forest alcançou 64,4% de acurácia, um resultado muito
próximo ao da Regressão Logística (64,6%) e apenas ligeiramente superior
ao baseline (63%). Isso mostra que, até o momento, os modelos testados
não estão conseguindo separar bem as classes, possivelmente devido ao
desbalanceamento ou à falta de variáveis mais discriminativas. Assim,
seria interessante explorar técnicas de balanceamento de classes,
ajustes de hiperparâmetros e a inclusão de algoritmos adicionais para
tentar melhorar a performance preditiva.
:::

::: {#QOOGh7SpGawR .cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="QOOGh7SpGawR" outputId="8cd00aa2-53be-44dc-8c15-5c245120faf9"}
``` python
# Random Forest - Treina um modelo de Random Forest (com padronização prévia) e avalia sua acurácia no conjunto de teste
pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
])
pipe_rf.fit(X_train, y_train)
y_pred_rf = pipe_rf.predict(X_test)
print("Acurácia Random Forest:", accuracy_score(y_test, y_pred_rf))
```

::: {.output .stream .stdout}
    Acurácia Random Forest: 0.643972357307397
:::
:::

::: {#w7sS41ZnGawS .cell .markdown id="w7sS41ZnGawS"}
## 4. Otimização de Hiperparâmetros {#4-otimização-de-hiperparâmetros}

O modelo de Random Forest obteve melhor resultado quando usamos árvores
limitadas a profundidade 10, com divisão mínima de 2 amostras e 100
árvores na floresta. Essa configuração mostrou ser o ponto de equilíbrio
entre complexidade e generalização, evitando tanto o underfitting quanto
o overfitting.
:::

::: {#j8f4SkqxGawS .cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="j8f4SkqxGawS" outputId="ccc1b6a5-207c-4c90-9cd2-e05ff64360a8"}
``` python
#Esse bloco faz a busca pelos melhores hiperparâmetros do Random Forest usando validação cruzada (GridSearchCV) com métrica F1, retornando o modelo otimizado

param_grid = {
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5]
}
grid = GridSearchCV(pipe_rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
print("Melhores parâmetros:", grid.best_params_)
best_model = grid.best_estimator_
```

::: {.output .stream .stdout}
    Melhores parâmetros: {'clf__max_depth': 10, 'clf__min_samples_split': 2, 'clf__n_estimators': 100}
:::
:::

::: {#BDeQy_sMGawT .cell .markdown id="BDeQy_sMGawT"}
## 5. Avaliação Final {#5-avaliação-final}

O modelo de Random Forest otimizado alcançou 66% de acurácia, melhorando
um pouco em relação ao baseline (63%). Ele tem bom desempenho em
identificar motores bons (classe 1, recall de 85%), mas falha bastante
em reconhecer motores ruins (classe 0, recall de apenas 32%), o que é
crítico no contexto do problema. A curva ROC com AUC = 0.688 confirma
que o modelo consegue distinguir as classes melhor que o acaso, mas
ainda de forma limitada.
:::

::: {#7Wfynz5nGawT .cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" id="7Wfynz5nGawT" outputId="8457ab9e-1e21-4866-f63e-30e1140e36bf"}
``` python
# Avalia o modelo otimizado no conjunto de teste, exibindo relatório de métricas, matriz de confusão e curva ROC com AUC.
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best))

cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()

fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1]):.3f}")
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curva ROC')
plt.legend()
plt.show()
```

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

               0       0.56      0.32      0.40      1444
               1       0.68      0.85      0.76      2463

        accuracy                           0.66      3907
       macro avg       0.62      0.58      0.58      3907
    weighted avg       0.64      0.66      0.63      3907
:::

::: {.output .display_data}
![](3bfd7573cfe22e40b4a08820222772b12f60345f.png)
:::

::: {.output .display_data}
![](c01fe5cbd417011bba1bc42bd6a886da73f7c335.png)
:::
:::
