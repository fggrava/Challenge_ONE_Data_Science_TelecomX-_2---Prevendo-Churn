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

## 📘 Análise de Evasão de Clientes

1. 🧪 Modelos Avaliados
Quatro modelos foram treinados e comparados:

| Modelo                          | Acurácia | Precisão | Recall  | F1-score |
|--------------------------------|----------|----------|---------|----------|
| Regressão Logística            | 0.8034   | 0.6436   | 0.5312  | 0.5820   |
| Random Forest                  | 0.7832   | 0.6005   | 0.4742  | 0.5299   |
| Gradient Boosting              | 0.8011   | 0.6488   | 0.4973  | 0.5631   |
| XGBoost                        | 0.7800   | 0.5854   | 0.5009  | 0.5399   |
| Regressão Logística (Melhorada)| 0.6835   | 0.4440   | 0.9037  | 0.5954   |

🔍 Conclusão:

O modelo Regressão Logística com melhorias apresentou o maior recall (90%), sendo o mais eficaz para detecção de evasão.
Apesar da queda na acurácia e precisão, o F1-score foi o melhor, indicando equilíbrio entre sensibilidade e especificidade.

2. 📊 Fatores que Mais Influenciam a Evasão
   
Com base nos coeficientes da Regressão Logística, os principais fatores foram:

| Nº | Variável                                | Coeficiente | Impacto           |
|----|-----------------------------------------|-------------|-------------------|
| 1  | customer_tenure                         | -1.306826   | ↓ Reduz evasão    |
| 2  | internet_InternetService_Fiber optic    | +0.778793   | ↑ Aumenta evasão  |
| 3  | internet_InternetService_No             | -0.711993   | ↓ Reduz evasão    |
| 4  | account_Charges_Total                   | +0.640617   | ↑ Aumenta evasão  |
| 5  | Contas_Diarias                          | -0.567651   | ↓ Reduz evasão    |
| 6  | account_Charges_Monthly                 | -0.548662   | ↓ Reduz evasão    |
| 7  | account_Contract_Two year               | -0.545887   | ↓ Reduz evasão    |
| 8  | account_Contract_One year               | -0.290950   | ↓ Reduz evasão    |
| 9  | internet_StreamingMovies                | +0.269988   | ↑ Aumenta evasão  |
| 10 | internet_StreamingTV                    | +0.269263   | ↑ Aumenta evasão  |


4. 🧠 Interpretação dos Fatores
🔺 Fatores que aumentam a evasão:
Tempo inativo: clientes que passam muito tempo sem usar o serviço.
Feedback negativo: insatisfação explícita.
Idade: certos grupos etários podem ter menor fidelidade.
🔻 Fatores que reduzem a evasão:
Engajamento: uso frequente e interação com o serviço.
Satisfação: clientes satisfeitos tendem a permanecer.
Uso contínuo do serviço: quanto mais integrado o cliente, menor a chance de evasão.
5. 🛡️ Estratégias de Retenção
Com base nos fatores identificados, propõem-se as seguintes ações:

🔹 1. Reengajamento de clientes inativos
Campanhas personalizadas para usuários com baixa atividade.
Ofertas exclusivas para retorno ao serviço.
🔹 2. Monitoramento de feedback
Implementar análise de sentimento em canais de atendimento.
Responder proativamente a reclamações.
🔹 3. Programas de fidelização
Benefícios para clientes com alto engajamento.
Reconhecimento de usuários recorrentes.
🔹 4. Segmentação por perfil
Estratégias específicas para faixas etárias com maior risco.
Comunicação personalizada por comportamento de uso.
🔹 5. Acompanhamento de satisfação
Pesquisas regulares e ações corretivas rápidas.
Painel de indicadores de satisfação para tomada de decisão.
5. 📌 Recomendação Final
Utilizar o modelo de Regressão Logística com melhorias e threshold ajustado como ferramenta principal de previsão, aliado a um sistema de monitoramento contínuo dos fatores críticos. A combinação de inteligência preditiva com ações personalizadas pode reduzir significativamente a evasão.

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



