# Challenge_ONE_Data_Science_TelecomX-_2---Prevendo-Churn
# 📊 Previsão de Evasão de Clientes (Churn Prediction)

Este projeto tem como objetivo prever a **evasão de clientes (churn)** em uma empresa de telecomunicações com base em variáveis comportamentais, contratuais e demográficas. A análise foi conduzida com técnicas de ciência de dados, aprendizado de máquina e visualização para identificar os principais fatores que influenciam a saída de clientes e propor estratégias de retenção.

---

## 🗂️ Estrutura do Projeto

---

## 🧹 Preparação dos Dados

### 🔢 Classificação das Variáveis

- **Categóricas**: tipo de contrato, método de pagamento, serviços adicionais, gênero, faixa etária.
- **Numéricas**: tempo de contrato, total gasto, número de serviços, satisfação, engajamento.

### 🔄 Etapas de Transformação

- **One-Hot Encoding** para variáveis categóricas com múltiplas categorias.
- **Label Encoding** para variáveis binárias.
- **StandardScaler** para normalização dos dados numéricos (média = 0, desvio padrão = 1).

### ✂️ Separação dos Dados

- Divisão em **70% treino / 30% teste** com `stratify=y` para manter a proporção de evasivos.
- Aplicação de **SMOTE** para balancear as classes e melhorar o recall.

---

## 🧠 Modelagem e Justificativas

### Modelos testados:

- **Regressão Logística** (com e sem melhorias)
- **Random Forest**
- **Gradient Boosting**
- **XGBoost**

### Escolhas estratégicas:

- **Regressão Logística com melhorias** foi o modelo final escolhido por apresentar o melhor **recall (90%)** e **F1-score**, sendo ideal para detectar evasão.
- Ajuste de **threshold** para melhorar a sensibilidade.
- Uso de **class_weight='balanced'** para compensar o desbalanceamento.

---

## 📈 Exemplos de Gráficos e Insights

- 🔥 **Heatmap de correlação**: mostrou forte relação negativa entre tempo de contrato e evasão.
- 📦 **Boxplot**: clientes com menor tempo de contrato e menor gasto total têm maior propensão à evasão.
- 📊 **Gráfico de coeficientes**: variáveis como tempo inativo, feedback negativo e engajamento foram os principais influenciadores.

---

## 🚀 Como Executar o Projeto

### 1. Instalar bibliotecas necessárias
### 2. Carregar os dados
       Coloque o arquivo TelecomX_Data_normalizado2.csv na pasta data/.
### 3. Executar o notebook
      Abra o notebook notebook/churn_analysis.ipynb e execute as células passo a passo.
### 4. Carregar o modelo treinado
      import pickle
      with open('modelo/modelo_treinado.pkl', 'rb') as file:
           modelo = pickle.load(file)
      # Fazer previsões
      y_pred = modelo.predict(X_novos_dados)

## 📌 Conclusão

Este projeto fornece uma base sólida para prever churn e tomar decisões estratégicas baseadas em dados. A combinação de análise exploratória, modelagem preditiva e interpretação de variáveis permite ações direcionadas para retenção de clientes.

## 🧠 Autor

Projeto desenvolvido por Fernando Grava como projeto final do programa ONE - ORACLE.



