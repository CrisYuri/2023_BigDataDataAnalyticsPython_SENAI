import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

nomes_colunas = ['Comprimento das sépalas', 'Largura das sépalas', 'Comprimento das pétalas', 'Largura das pétalas', 'Tipos de Flores']

dataframe = pd.read_csv('/Users/crisyuriok/Library/CloudStorage/OneDrive-Personal/1-BigDataDataAnalyticsPyhton/Aula06.23.11.25/DataBase_Iris/iris.data',
                        sep = ',',
                        names = nomes_colunas)

print(dataframe.head())

boxplot = sns.boxplot(data=dataframe,
                      x='Tipos de Flores',
                      y='Comprimento das sépalas',
                      palette = 'pastel',
                      )
boxplot.set_title('Boxplot da base de dados Íris')

plt.show()
