# README — Classificação da Condição do Motor

Este notebook serve como **README / guia de execução** do projeto. Ele contém uma breve descrição das etapas e o código completo para reproduzir a análise e os modelos usados (baseline, regressão logística, Random Forest, otimização por GridSearch, SMOTE, XGBoost e LightGBM), exatamente conforme o script final fornecido.

### Como usar
1. Faça upload deste notebook no Google Colab.
2. Execute as células em ordem (ou `Runtime > Run all`).
3. O notebook baixa o dataset direto do repositório e gera saídas e gráficos.

Se quiser, copie e cole partes do código no seu notebook principal — aqui o objetivo é servir como documentação e guia para quem for rodar o projeto.

## 1) Configuração inicial e imports
As bibliotecas abaixo são necessárias para rodar o projeto. Se alguma não estiver instalada (ex.: `xgboost`, `lightgbm`, `imblearn`), instale via `pip install` no Colab.

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

```python
baseline = DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE)
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)
print('\n--- Baseline ---')
print('Acurácia:', accuracy_score(y_test, y_pred_base))
print(classification_report(y_test, y_pred_base))
```

## 6) Modelos iniciais: Regressão Logística e Random Forest

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

## 10) Salvar modelo final

```python
joblib.dump(best_model, 'best_model_engine.pkl')
print('\n✅ Modelo salvo: best_model_engine.pkl')
```

