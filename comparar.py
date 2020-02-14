import numpy as np
import matplotlib.pyplot as plt

#Definimos el prior con las regiones en las que han de estar los parametros
def logprior(v):
    for i in range(len(v)):
        if v[i] > 0 and v[i] < 100:
            p = 1.0
        else:
            p= 0
            break
    return p

#El modelo matematico que sigue la caida libre de un objeto con una velocidad inicial
def modelo1(v, x):
    n_dim = len(v)-1
    modelo1 = 0
    for i in range(n_dim):
        modelo1 += v[i]*x**(i)
    return modelo1

def modelo2(v, x):
    modelo2 = v[0]*(np.exp(-0.5*(x-v[1])**2/v[2]**2))
    return modelo2

#Comparacion de la propuesta con los datos experimentales en logaritmo
def loglikelihood(y, x, v1,v2, sigmas):
    L = np.zeros(len(y))
    P = np.zeros(len(y))
    for i in range(len(y)):
        L += np.log(1.0/np.sqrt(2.0*np.pi*sigmas[i]**2))+(-0.5*(modelo1(v1,x[i])-y[i])**2/(sigmas[i]**2))
        P += np.log(1.0/np.sqrt(2.0*np.pi*sigmas[i]**2))+(-0.5*(modelo2(v2,x[i])-y[i])**2/(sigmas[i]**2))
    return L , P

def logposterior(L,P,y):
    post =  L+P
    evidencia = np.trapz(np.exp(post), y)
    logpost = post - evidencia
    return  logpost

#Datos observados
data = np.loadtxt("data_to_fit.txt", skiprows=1)
Y = data[:,1]
X = data[:,0]
Sigmas = data[:,2]

#Numero con el que se realizara el MCMH
N = 10000
lista_v1 = [[1,1,1]]
lista_v2 = [[1,1,1]]
sigma_v=0.01
for i in range(1,N):
    #Proponemos un nuevo beta en funcion del anterior mas una distribucion normal
    propuesta_v1  = lista_v1[i-1] + np.random.normal(loc=0.0, scale=sigma_v, size=3)
    propuesta_v2  = lista_v2[i-1] + np.random.normal(loc=0.0, scale=sigma_v, size=3)
    
    MV1, MV2 = loglikelihood(Y,X,lista_v1[i-1],lista_v2[i-1],Sigmas)
    MN1, MN2 = loglikelihood(Y,X,propuesta_v1,propuesta_v2,Sigmas)
    #Se crean los Posteriors nuevo, y viejo con los anteriores, con el fin de tener el criterio de comparacion
  
    logposterior1_viejo = logposterior(MV1,logprior(lista_v1[i-1]),Y)
    logposterior1_nuevo = logposterior(MN1,logprior(propuesta_v1),Y)   
    
    logposterior2_viejo = logposterior(MV2,logprior(lista_v2[i-1]),Y)
    logposterior2_nuevo = logposterior(MN2,logprior(propuesta_v2),Y)  
    
    #criterio de comparacion
    r1 = min(np.exp(logposterior1_nuevo-logposterior1_viejo))
    r2 = min(np.exp(logposterior2_nuevo-logposterior2_viejo))
    alpha = np.random.random()
    
    #indexamos la propuesta 1
    if(alpha<r1):
        lista_v1.append(propuesta_v1)
    #No se indexa e indexamos el anterior
    else:
        lista_v1.append(lista_v1[i-1])
    #indexamos la propuesta 2
    if(alpha<r2):
        lista_v2.append(propuesta_v2)
    #No se indexa e indexamos el anterior
    else:
        lista_v2.append(lista_v2[i-1])
#Convertimos el todo en arrays para poder graficarlo en el histograma
lista_v1 = np.array(lista_v1)
lista_v2 = np.array(lista_v2)

#Construyamos los histagramas de cada uno de los cinco betas
plt.figure(figsize=(20, 15))
for i in range(3):
    plt.subplot(3,3,i+1)
    a, b, c = plt.hist(lista_v1[:,i], bins=50, density=True)
    bin_max = np.where(a == a.max())
    desv = np.std(lista_v1[:,i])
    mB = np.mean(lista_v1[:,i])
    plt.title(r"Distribucion del M1 $\beta_{:.0f}$. Con un valor medio de {:.2f} $\pm$ {:.2f}".format(float(i) , float(mB), float(desv)))
    plt.xlabel(r"$\beta_{:.0f}$".format(float(i)))
    
for i in range(3):
    plt.subplot(3,3,i+4)
    a, b, c = plt.hist(lista_v2[:,i], bins=50, density=True)
    bin_max = np.where(a == a.max())
    desv = np.std(lista_v2[:,i])
    mB = np.mean(lista_v2[:,i])
    plt.title(r"Distribucion del M2 $\beta_{:.0f}$. Con un valor medio de {:.2f} $\pm$ {:.2f}".format(float(i) , float(mB), float(desv)))
    plt.xlabel(r"$\beta_{:.0f}$".format(float(i)))
    
plt.savefig('Comparar.png')