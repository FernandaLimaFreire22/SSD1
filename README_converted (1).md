# README — Classificação da Condição do Motor

O objetivo deste projeto é desenvolver um Mínimo Produto Viável (MVP) para um problema de classificação, aplicando técnicas de machine learning em um contexto de manutenção preditiva.

A tarefa consiste em prever a condição de motores (0 = ruim, 1 = bom) com base em variáveis medidas por sensores, como pressão do óleo, pressão do combustível, pressão do fluido de arrefecimento, temperaturas e rotações por minuto (RPM).

Esse tipo de solução pode auxiliar empresas na tomada de decisões de manutenção, reduzindo falhas inesperadas e otimizando custos operacionais.

Hipótese

As variáveis coletadas pelos sensores dos motores contêm informações suficientes para prever corretamente se o motor se encontra em boa condição (1) ou em má condição (0).

## 1) Configuração inicial e imports
As bibliotecas abaixo são necessárias para rodar o projeto.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print('✅ Bibliotecas importadas com sucesso!')
```

## 2) Carregamento e definição do problema
O dataset é carregado diretamente do repositório e a variável alvo é `Engine Condition`.

```python
url = 'https://raw.githubusercontent.com/FernandaLimaFreire22/SSD1/main/engine_data2.csv'
df = pd.read_csv(url, sep=';')

print('\nAmostra do Dataset Original:')
print(df.head())
print('\nFormato (linhas, colunas):', df.shape)
print('\nTipos das colunas:\n', df.dtypes)

# Converter colunas que vieram como string com separador errado
cols_to_convert = ['Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
for col in cols_to_convert:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    df[col] = pd.to_numeric(df[col])

print('\nDescrição estatística:')
display(df.describe().T)

print('\nDistribuição da variável alvo (Engine Condition):')
print(df['Engine Condition'].value_counts(normalize=True) * 100)
```

## 3) Análise exploratória (gráficos rápidos)

Esses gráficos permitem visualizar como cada variável numérica do motor está distribuída. Observamos que algumas variáveis, como Engine rpm, apresentam uma distribuição mais concentrada, enquanto outras, como pressões e temperaturas, têm distribuições multimodais, sugerindo que existem diferentes condições de operação ou registros anômalos. Essa análise é importante porque ajuda a identificar padrões, possíveis outliers e diferenças de comportamento entre regimes de funcionamento do motor, o que pode impactar na detecção de falhas ou na previsão da condição do motor.

A matriz de correlação mostra que, no conjunto de dados, a maior parte das variáveis não apresenta relação linear significativa entre si. A única correlação mais perceptível é entre a rotação do motor (Engine rpm) e a condição do motor (Engine Condition), que aparece como uma correlação negativa fraca (-0,27). Isso sugere que, conforme a rotação aumenta, a condição do motor pode tender a se deteriorar levemente, enquanto as demais variáveis analisadas se comportam de forma praticamente independente.

```python
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
if 'Engine Condition' in num_cols:
    num_cols.remove('Engine Condition')

plt.figure(figsize=(14,8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2,3,i)
    sns.histplot(df[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Matriz de correlação')
plt.show()
```

## 4) Preparação dos dados e split
O dataset foi dividido em dois subconjuntos: 80% dos dados (15.628 registros) foram reservados para treinar o modelo e 20% (3.907 registros) para testá-lo. Essa separação garante que o modelo seja avaliado em dados que ele nunca viu, permitindo verificar sua capacidade de generalização. Além disso, o uso de estratificação assegura que a distribuição das classes da variável alvo seja mantida tanto no treino quanto no teste.

```python
print('Duplicados:', df.duplicated().sum())
df = df.drop_duplicates()

X = df.drop('Engine Condition', axis=1)
y = df['Engine Condition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print('Tamanho treino:', X_train.shape, ' | teste:', X_test.shape)
print('Distribuição treino:', y_train.value_counts(normalize=True))
print('Distribuição teste:', y_test.value_counts(normalize=True))
```

## 5) Baseline (DummyClassifier)
O modelo baseline alcançou 63% de acurácia simplesmente por prever sempre a classe mais frequente (classe 1). No entanto, ele falhou totalmente em identificar a classe 0, resultando em precisão, recall e f1-score iguais a 0 para essa categoria. Isso mostra que, apesar da acurácia parecer razoável, o modelo não é útil para distinguir as classes. Assim, o baseline serve apenas como referência inicial para compararmos modelos mais sofisticados, que precisam superar esse desempenho trivial.

```python
baseline = DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE)
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)
print('\n--- Baseline ---')
print('Acurácia:', accuracy_score(y_test, y_pred_base))
print(classification_report(y_test, y_pred_base))
```

## 6) Modelos iniciais: Regressão Logística e Random Forest
Foi treinado um modelo de Regressão Logística, que é um algoritmo de classificação linear. O pipeline aplicou uma padronização das variáveis antes do treino e, em seguida, o modelo foi avaliado no conjunto de teste. Ele alcançou 64,6% de acurácia, superando o baseline de 63%, o que indica que conseguiu extrair alguma informação útil dos dados. No entanto, como o ganho foi pequeno, isso sugere a necessidade de testar algoritmos mais robustos (como Random Forest ou XGBoost) e aplicar técnicas para lidar com o desbalanceamento de classes, a fim de melhorar a capacidade preditiva do modelo.

O modelo de Random Forest alcançou 64,4% de acurácia, um resultado muito próximo ao da Regressão Logística (64,6%) e apenas ligeiramente superior ao baseline (63%). Isso mostra que, até o momento, os modelos testados não estão conseguindo separar bem as classes, possivelmente devido ao desbalanceamento ou à falta de variáveis mais discriminativas. Assim, seria interessante explorar técnicas de balanceamento de classes, ajustes de hiperparâmetros e a inclusão de algoritmos adicionais para tentar melhorar a performance preditiva.

```python
# Regressão Logística
pipe_log = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=42, max_iter=1000))
])
pipe_log.fit(X_train, y_train)
y_pred_log = pipe_log.predict(X_test)
print('\n--- Logistic Regression ---')
print(classification_report(y_test, y_pred_log))

# Random Forest
pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
])
pipe_rf.fit(X_train, y_train)
y_pred_rf = pipe_rf.predict(X_test)
print('\n--- Random Forest ---')
print(classification_report(y_test, y_pred_rf))

# Importância das variáveis
importances = pipe_rf.named_steps['clf'].feature_importances_
feat_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,4))
sns.barplot(x=feat_importance.values, y=feat_importance.index)
plt.title('Importância das variáveis - Random Forest')
plt.show()
```

## 7) Otimização (GridSearch) para Random Forest
O Random Forest otimizado conseguiu uma boa taxa de acerto para os motores bons (classe 1), mas teve dificuldade em identificar corretamente os motores ruins (classe 0). Isso aconteceu porque os dados estão desbalanceados (muito mais exemplos de motores bons do que ruins), e o modelo tende a “aprender” mais sobre a classe majoritária.

```python
param_grid = {
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5]
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(pipe_rf, param_grid, cv=cv, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
print('\nMelhores parâmetros:', grid.best_params_)
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)
print('\n--- Random Forest Otimizado ---')
print(classification_report(y_test, y_pred_best))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - Melhor Modelo')
plt.show()
```

## 8) Balanceamento e modelos avançados (SMOTE + XGBoost/LightGBM)
O log do LightGBM mostra que, após aplicarmos o SMOTE, as classes ficaram balanceadas (aprox. 50% bons e 50% ruins). O modelo então inicia o treinamento de forma justa entre as classes e escolhe automaticamente a melhor forma de paralelizar os cálculos para acelerar o processo. Isso confirma que o balanceamento foi aplicado corretamente e que o algoritmo está rodando como esperado.

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Random Forest + SMOTE
pipe_rf_bal = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
])
pipe_rf_bal.fit(X_train, y_train)

# XGBoost + SMOTE
pipe_xgb = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('clf', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])
pipe_xgb.fit(X_train, y_train)

# LightGBM + SMOTE
pipe_lgbm = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('clf', LGBMClassifier(random_state=42))
])
pipe_lgbm.fit(X_train, y_train)
```

## 9) Comparação entre modelos com validação cruzada
A comparação mostrou que a Regressão Logística foi o modelo mais eficiente, superando inclusive técnicas mais complexas como Random Forest, XGBoost e LightGBM com SMOTE. O melhor F1 médio foi 0.759, enquanto o pior foi 0.669, uma diferença significativa. Isso indica que os algoritmos mais avançados não conseguiram superar o modelo simples e que ainda existe espaço para melhorar, seja ajustando melhor os hiperparâmetros, testando outras formas de balanceamento ou explorando mais variáveis.

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

modelos = {
    'Logistic Regression': pipe_log,
    'Random Forest (GridSearch)': best_model,
    'Random Forest + SMOTE': pipe_rf_bal,
    'XGBoost + SMOTE': pipe_xgb,
    'LightGBM + SMOTE': pipe_lgbm
}
resultados = {}
for nome, modelo in modelos.items():
    scores = cross_val_score(modelo, X, y, cv=cv, scoring='f1')
    resultados[nome] = [scores.mean(), scores.std()]

resultados_df = pd.DataFrame(resultados, index=['F1 médio', 'Desvio Padrão']).T
print('\n===== Comparação entre modelos =====')
print(resultados_df)

plt.figure(figsize=(8,5))
sns.barplot(x=resultados_df.index, y=resultados_df['F1 médio'])
plt.xticks(rotation=30)
plt.ylabel('F1 médio (validação cruzada)')
plt.title('Comparação de Modelos')
plt.show()

melhor = resultados_df['F1 médio'].max()
pior = resultados_df['F1 médio'].min()
print('\nResumo final:')
print(f"Melhor modelo teve F1 médio de {melhor:.3f}")
print(f"Pior modelo teve F1 médio de {pior:.3f}")
if melhor - pior < 0.05:
    print("👉 Todos os modelos ficaram muito próximos → limite está nos dados.")
else:
    print("👉 Houve diferença significativa entre os modelos → ainda há espaço para melhorar.")
```

O modelo otimizado apresenta um desempenho muito bom nos dados de treino (77,6%), mas seu desempenho cai para 65,6% nos dados de teste. A diferença de aproximadamente 12 pontos indica que o modelo pode estar sofrendo overfitting, ou seja, está ajustado demais aos dados de treino e não generaliza tão bem para dados novos. Para melhorar, podemos considerar técnicas como regularização, poda do modelo, coleta de mais dados ou validação cruzada para reduzir o overfitting.

## 10) Salvar modelo final

```python
joblib.dump(best_model, 'best_model_engine.pkl')
print('\n✅ Modelo salvo: best_model_engine.pkl')
```

