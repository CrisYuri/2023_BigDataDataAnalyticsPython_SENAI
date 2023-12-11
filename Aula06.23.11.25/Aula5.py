'''for i in range(20):
    print(f"Inicializando a iteração -> {i}")
    if i > 8:
        print(f"Finalizando a instrução {i}")
        break
        print("Ação nunca vista")
    if(i > 4):
        print(f"Continuando a iteração -> {i}")
        continue
        print("Ação nunca vista")
    print(f"Loop normal -> iteração {i}")
    print(f"i^2 é {i**2}")

print("Próxima instrução...")'''

import matplotlib.pyplot as plt

produtos = ['Soja em grão', 'Cana de Açúcar', 'Milho em grão', 'Mandioca']

quantidade_pct = (33.6, 16.8, 10.3, 6.2, 3.8)

plt.figure(figsize=(12, 6))
plt.bar(produtos, quantidade_pct, color='green')

# Inserindo nome nos eixos e gráfico
plt.ylabel('Produtos Agrícolas')
plt.xlabel('Participação do produto [%]')
plt.title('5 Produtos com maior participação da industria agrícola')

plt.show()