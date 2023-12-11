import numpy as np
import statistics as st
import matplotlib.pyplot as plt



M = np.loadtxt('treinamento.txt')
# M = pd.read_csv('treinamento.txt', sep='\t', encoding='UTF-8', header=None)
# M = np.array(df)


# Organizando Matriz dados
T = -1*np.ones((M.shape[0], M.shape[1])) # cria matriz T já com -1, com tamanho de M.shape[0] - referente ao numero de linhas e M.shape[1] - ref ao numero de colunas
T[:,1:] = M[:,:-1] # sobrepõe os dados da M na T, sem a última coluna da M, que é o resultado esperado (0 ou 1) e a partir da coluna 2 da T, pois a coluna 1 é o limiar (-1)

d = M[:,-1:] # matriz dos valores desejados, extraída da ultima coluna da matriz original

# Peso sinaptico aleatorio
w = np.zeros(M.shape[1]) # shape de colunas
w[0] = -1 # preenche a primeira coluna do w com -1
w[1:] = np.random.rand(3) # preenche as outras 3 colunas com 3 números aleatórios
wi = w; # peso inicial


print(f'w inicial: ' + str(wi))

n = 0.01 # estipulando valor de neta taxa de aprendizado
epoca = 0 # iniciando o contador de épocas

# função que vai testar se a saída é igual a esperada ou se é diferente da esperada e será necessário recalcular o W
def FuncDegrau(q):
    g = np.zeros(len(q))
    for k in range(len(q)):
        if q[k] >= 0:
            g[k] = 1
        else:
            g[k] = -1
    return g

# Inicio da fase de treinamento
erro = True
while erro == True:

    erro = False

    epoca = epoca+1

    for k in range(M.shape[0]):
        x = T[k,:] # extrai da matriz T as colunas uma a uma, para testar com o W
        u = [w @ x] # faz teste de W x a coluna do T e resulta na saida u

        y= FuncDegrau(u) # testa a saída

        if y != d[k]: # Recalcula
            w = w + (n*(d[k]-y)*x).T
            erro = True
    print('Época:' + str(epoca) + '-> W: ' + str(w))
    if epoca == 1000: # Limite das épocas
        break # Força a saída do laço

print('w final: ' + str(w))
print('Época para convergência: ' + str(epoca))

# Porcentagem de acerto e plotagem da comparação
u = w @ T.T 
outt = np.zeros(u.shape[0])

for k in range(u.shape[0]):
    outt[k] = FuncDegrau(np.array([u[k]]))

result_Tr = d.T - outt.T

errM = st.mean(abs(result_Tr.flatten()))*100

M_Te = np.loadtxt('teste.txt')

print(f'Porcentagem de acerto Treinamento: {100-errM:.2f} %')

plt.figure(figsize=(5,5),dpi=150)
plt.plot(outt,marker="o", color='red', linestyle='None', markerfacecolor='None' )
plt.plot(d, marker="+", color='blue', linestyle='None', markerfacecolor='None')
plt.legend(('Saída Rede', 'Desejado'))
plt.title('Saída X Desejado - Treinamento')
plt.show()

# 

# Organizando Matriz dados
T_Te = -1*np.ones((M_Te.shape[0], M_Te.shape[1])) # cria matriz T já com -1, com tamanho de M.shape[0] - referente ao numero de linhas e M.shape[1] - ref ao numero de colunas
T_Te[:,1:] = M_Te[:,:-1] # sobrepõe os dados da M na T, sem a última coluna da M, que é o resultado esperado (0 ou 1) e a partir da coluna 2 da T, pois a coluna 1 é o limiar (-1)

d_Te = M_Te[:,-1:] # matriz dos valores desejados, extraída da ultima coluna da matriz original

y_Te = np.zeros(M_Te.shape[0])
y_Tebar = np.zeros(M_Te.shape[0]) # saida para poder plotar
out = np.zeros(M_Te.shape[0]) # saida do array de u


for k in range(M_Te.shape[0]):
    x = T_Te[k,:]
    u = [w @ x]

    out[k] = np.array([u])
    y_Te[k] = FuncDegrau(u)

    if (y_Te[k] == 1): # conversão para classes
        print('Classe C1')
        y_Tebar[k] = 1
    else:
        print('Classe C2')
        y_Tebar[k] = -1
# print(out)

# Porcentagem de acerto e plotagem da comparação
result_Te = d_Te.T - y_Te.T

errT = st.mean(abs(result_Te.flatten()))*100

M_Te = np.loadtxt('teste.txt')

print(f'Porcentagem de acerto Teste: {100-errT:.2f} %')

plt.figure(figsize=(5,5),dpi=150)
plt.plot(y_Te,marker="o", color='red', linestyle='None', markerfacecolor='None' )
plt.plot(d_Te, marker="+", color='blue', linestyle='None', markerfacecolor='None')
plt.legend(('Saída Rede', 'Desejado'), loc = 'center right')
plt.title('Saída X Desejado - Teste')
plt.show()

fig, ax = plt.subplots(figsize=(8,5),dpi=150)
ax.bar(np.array(list(range(T_Te.shape[0]))),y_Tebar)
ax.set_xticks(np.array(list(range(T_Te.shape[0]))))
ax.set_xticklabels(np.array(list(range(1,T_Te.shape[0]+1))))
ax.set_yticks([-1, 1])
ax.set_yticklabels(["Classe B", "Classe A"])
ax.set_xlabel("Amostra")
ax.set_ylabel("Classificação")
plt.show()
