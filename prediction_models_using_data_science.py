#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#- Instalar dependencias
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install scikit-learn')


# # Projeto Ciencia de dados - Previsão de Vendas
# 
# - Nosso desafio é conseguir prever as vendas que vamos ter em um determinado periodo com base nos gastos em anuncios das 3 grandes redes em que a empresa hashtag investe: TV, Jornal e rádio
# - Base de dados Adversiting.csv

# # Passo a passo de um projeto de ciencia de dados
# 
#  - Passo 1: Entendimento do desafio
#  - Passo 2: Entendimento da área/empresa
#  - Passo 3: Extração/obtenção de dados
#  - Passo 4: Ajuste de dados (Tratamento de dados)
#  - Passo 5: Analise exploratoria
#  - Passo 6: Modelagem / Algoritmos (Aqui entra a inteligencia artificial se necessário)
#  - Passo 7: Interpretação dos resultados

# In[3]:


# Importar biblioteca pandas
import pandas as pd

# importar baee de dados
tabela = pd.read_csv(r"D://Intensivao de Python/aula4/advertising.csv")

display(tabela)


# In[4]:


# Descrobir a correlação dentro da tabela por meio de graficos
# existem 3 grandes libs de graficos: plotly - matplotlib - seaborn
# importar as libs para o projeto

import seaborn as sns
import matplotlib.pyplot as plt

# sempre que for criar o graficos no python, deve se fazer em 2 etapas: 1 - criar o grafico, 2 - exibir o grafico
# - Criar o grafico (o pairplot apenas existe no seaborn)
sns.pairplot(tabela)

# - exibir o grafico (geramos com o seaborn, mas vamos exibir com o plt)
plt.show()


# In[5]:


# - Criar o grafico heatmap (mapa de calor)
#sns.heatmap(tabela.corr()) # cria o mapa usando os dados de correlação da tabela
#sns.heatmap(tabela.corr(), cmap="Wistia") # cria o mapa usando o esquema de cores Wistia
sns.heatmap(tabela.corr(), cmap="Wistia", annot=True) # cria o mapa usando o esquema de cores Wistia e usando as anotações dos dados
# - exibir o grafico (geramos com o seaborn, mas vamos exibir com o plt)
plt.show()


# In[19]:


# separar os dados em treino e teste, para isso, importar as libs do sklearn
from sklearn.model_selection import train_test_split

# separar os dados em X e y aonde y = quem eu quero calcular e x é tudo, menos o Y
y = tabela["Vendas"]
#x = tabela[["TV", "Radio", "Jornal"]] # usar 2 colchetes para informar varias tabelas de uma só vez
x = tabela.drop("Vendas", axis=1) # Nesse modelo, temos todas as colunas, removendo apenas a coluna Vendas

# os dados para teste podem usar o valor padrão do train test, porém do jeito abaixo, podemos especificar o valor de teste
# por exemplo uma margem de 70 / 30, 80 / 20 ... o recomendado é 70% treino e 30% teste

# x_treino, y_treino, x_teste, y_teste = train_test_split(x, y, teste_size=0.3) <== aqui especifico o tamanho do teste (30%)
# x_treino, y_treino, x_teste, y_teste = train_test_split(x, y)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)


# # temos um problema de regressão - Vamos escolher os modelos que iremos usar:
# 
# - Regressão linear
# - RandomForest (arvore de decisão)

# In[21]:


# como que a gente cria uma IA: importa, cria e treina a inteligencia
# importa
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# cria a IA
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# treina a IA
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)


# # Teste de IA e Avaliação do melhor modelo
# 
# - Vamos usar o R² -> diz o % que o nosso modelo consegue explicar o que acontece

# In[24]:


# importar o r2_score para calcular o R²
from sklearn.metrics import r2_score

#crio as previsoes
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

# compara as previsoes com o gabarito
print(r2_score(y_teste, previsao_regressaolinear))
print(f"{r2_score(y_teste, previsao_arvoredecisao):.1%}") ## exibe formatado em percentual


# In[ ]:


# O melhor modelo é o modelo de arvore de decisão


# # Visualização Gráfica das Previsões

# In[28]:


tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsoes ArvoreDecisao"] = previsao_arvoredecisao
tabela_auxiliar["Previsoes Regressao Linear"] = previsao_regressaolinear

plt.figure(figsize=(15,6))
sns.lineplot(data=tabela_auxiliar)
plt.show()


# # Qual a importancia de cada variavel para as vendas?

# In[27]:


sns.barplot(x=x_treino.columns, y=modelo_arvoredecisao.feature_importances_)
plt.show()

# Caso queira comparar Radio com Jornal
# print(df[["Radio", "Jornal"]].sum())


# In[29]:


# Como fazer uma nova previsao
# Digamos que o chefe peça novas previões - como proceder?
# importar a nova_tabela com o pandas (a nova tabela tem que ter os dados de TV, Radio e Jornal)
# previsao = modelo_randomforest.predict(nova_tabela)
# print(previsao)

# informa os novos valores, importando a nova base de dados
novos_valores = pd.read_csv(r"D://Intensivao de Python/aula4/novos.csv")

display(novos_valores)

nova_previsao = modelo_arvoredecisao.predict(novos_valores)

print(nova_previsao)


# In[ ]:




