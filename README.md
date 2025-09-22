MVP – Ciência de Dados: Classificação da Condição de Motores

Discente: Fernanda Lima
Data: 17/09/2025

1. Definição do Problema

O objetivo deste projeto é desenvolver um Mínimo Produto Viável (MVP) para um problema de classificação, aplicando técnicas de machine learning em um contexto de manutenção preditiva.

A tarefa consiste em prever a condição de motores (0 = ruim, 1 = bom) com base em variáveis medidas por sensores, como pressão do óleo, pressão do combustível, pressão do fluido de arrefecimento, temperaturas e rotações por minuto (RPM).

Esse tipo de solução pode auxiliar empresas na tomada de decisões de manutenção, reduzindo falhas inesperadas e otimizando custos operacionais.

Hipótese

As variáveis coletadas pelos sensores dos motores contêm informações suficientes para prever corretamente se o motor se encontra em boa condição (1) ou em má condição (0).

Dataset

Fonte: Dataset interno disponibilizado para o projeto.

Formato: Arquivo CSV contendo 7 colunas e 19.535 registros.

Atributos (features):

Engine rpm

Lub oil pressure

Fuel pressure

Coolant pressure

Lub oil temp

Coolant temp

Variável alvo: Engine Condition (0 = ruim, 1 = bom)

Configuração do Ambiente
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Configuração dos gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("Ambiente configurado com sucesso!")

2. Preparação dos Dados

Nesta etapa, o dataset é carregado, limpo e transformado para estar apto à modelagem.

Operações Realizadas

Carga dos Dados: leitura do CSV hospedado no GitHub.

Tratamento de Formatação: muitas colunas numéricas continham pontos (.) como separador de milhar. Esses caracteres foram removidos e os valores convertidos para tipo float.

Análise Exploratória: estatísticas descritivas, histogramas e matriz de correlação para entender relações entre variáveis.

Remoção de Duplicados: linhas repetidas foram eliminadas.

Separação de Variáveis: definição de X (features) e y (alvo).

Divisão em Conjuntos: divisão em 80% treino e 20% teste, preservando a proporção das classes (estratificação).

Implementação
2.1. Carga e limpeza dos dados
url = 'https://raw.githubusercontent.com/FernandaLimaFreire22/SSD1/main/engine_data2.csv'
df = pd.read_csv(url, sep=';')

# Correção de colunas numéricas com separador de milhar
cols_to_convert = ['Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 
                   'lub oil temp', 'Coolant temp']
for col in cols_to_convert:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    df[col] = pd.to_numeric(df[col])
