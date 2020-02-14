import numpy as np
import matplotlib.pyplot as plt

#Definimos el prior con las regiones en las que han de estar los parametros
def logprior(v):
    for i in range(len(v)):
        if v[i] > -10 and v[i] < 10:
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
def modelo3(v,x):
    modelo3 = v[0]*(np.exp(-0.5*(x-v[1])**2/v[2]**2))
    modelo3 += v[0]*(np.exp(-0.5*(x-v[3])**2/v[4]**2))
    return modelo3
#Comparacion de la propuesta con los datos experimentales en logaritmo
def loglikelihood(y, x, v1,v2,v3 ,sigmas):
    L = np.zeros(len(y))
    P = np.zeros(len(y))
    Q = np.zeros(len(y))
    for i in range(len(y)):
        L += np.log(1.0/np.sqrt(2.0*np.pi*sigmas[i]**2))+(-0.5*(modelo1(v1,x[i])-y[i])**2/(sigmas[i]**2))
        P += np.log(1.0/np.sqrt(2.0*np.pi*sigmas[i]**2))+(-0.5*(modelo2(v2,x[i])-y[i])**2/(sigmas[i]**2))
        Q += np.log(1.0/np.sqrt(2.0*np.pi*sigmas[i]**2))+(-0.5*(modelo3(v3,x[i])-y[i])**2/(sigmas[i]**2))
    return L , P ,Q

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
N = 100
lista_v1 = [[1,1,1]]
lista_v2 = [[1,1,1]]
lista_v3 = [[1,1,1,1,1]]

sigma_v1 = 0.01
sigma_v2 = 0.005
sigma_v3 = 0.005
for i in range(1,N):
    #Proponemos un nuevo beta en funcion del anterior mas una distribucion normal
    propuesta_v1  = lista_v1[i-1] + np.random.normal(loc=0.0, scale=sigma_v1, size=3)
    propuesta_v2  = lista_v2[i-1] + np.random.normal(loc=0.0, scale=sigma_v2, size=3)
    propuesta_v3  = lista_v3[i-1] + np.random.normal(loc=0.0, scale=sigma_v3, size=5)
    
    MV1, MV2, MV3 = loglikelihood(Y,X,lista_v1[i-1],lista_v2[i-1],lista_v3[i-1],Sigmas)
    MN1, MN2, MN3 = loglikelihood(Y,X,propuesta_v1,propuesta_v2,propuesta_v3,Sigmas)
    #Se crean los Posteriors nuevo, y viejo con los anteriores, con el fin de tener el criterio de comparacion
  
    logposterior1_viejo = logposterior(MV1,logprior(lista_v1[i-1]),Y)
    logposterior1_nuevo = logposterior(MN1,logprior(propuesta_v1),Y)   
    
    logposterior2_viejo = logposterior(MV2,logprior(lista_v2[i-1]),Y)
    logposterior2_nuevo = logposterior(MN2,logprior(propuesta_v2),Y)  
    
    logposterior3_viejo = logposterior(MV3,logprior(lista_v3[i-1]),Y)
    logposterior3_nuevo = logposterior(MN3,logprior(propuesta_v3),Y)
    
    #criterio de comparacion
    r1 = min(np.exp(logposterior1_nuevo-logposterior1_viejo))
    r2 = min(np.exp(logposterior2_nuevo-logposterior2_viejo))
    r3 = min(np.exp(logposterior3_nuevo-logposterior3_viejo))
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
    #indexamos la propuesta 3
    if(alpha<r3):
        lista_v3.append(propuesta_v3)
    #No se indexa e indexamos el anterior
    else:
        lista_v3.append(lista_v3[i-1])
        
mitad = int(N/2)
print(mitad)
#Convertimos el todo en arrays para poder graficarlo en el histograma
lista_v1 = lista_v1[mitad:]
lista_v1 = np.array(lista_v1)
lista_v2 = lista_v2[mitad:]
lista_v2 = np.array(lista_v2)
lista_v3 = lista_v3[mitad:]
lista_v3 = np.array(lista_v3)

#fig1, axesArray1 = plt.subplots(nrows=1,ncols=4,figsize=(20, 20))
#fig1.subplots_adjust(hspace=.5,wspace=0.4)

#bestparamM1 = []
#bestparamSigmasM1 = []
#for i in range(3):
#    axis1 = axesArray1[1, i+1]
#    a, b, c = axis1.hist(lista_v1[:,i], bins=50, density=True)
#    bin_max = np.where(a == a.max())
#    desv =np.std(lista_v1[:,i])
#    mB =np.mean(lista_v1[:,i])
#    bestparamSigmasM1.append(desv)
#    bestparamM1.append(mB)
#    axis1.set_title(r"Distribucion del M1 $\beta_{:.0f}$. Con un valor medio de {:.2f} $\pm$ {:.2f}".format(float(i) , float(mB), float(desv)))
#    axis1.set_xlabel(r"$\beta_{:.0f}$".format(float(i)))


#fig2, axesArray2 = plt.subplots(nrows=1,ncols=4,figsize=(20, 20))
#fig2.subplots_adjust(hspace=.5,wspace=0.4)
#bestparamM2 = []
#bestparamSigmasM2 = []
#for i in range(3):
#    axis2 = axesArray2[1, i+1]
#    a, b, c = axis2.hist(lista_v2[:,i], bins=50, density=True)
##    bin_max = np.where(a == a.max())
 #   desv =np.std(lista_v2[:,i])
 #   mB =np.mean(lista_v2[:,i])
 #   bestparamSigmasM2.append(desv)
 #   bestparamM2.append(mB)
 #   axis2.set_title(r"Distribucion del M2 $\beta_{:.0f}$. Con un valor medio de {:.2f} $\pm$ {:.2f}".format(float(i) , float(mB), float(desv)))
 #   axis2.set_xlabel(r"$\beta_{:.0f}$".format(float(i)))
    
#fig3, axesArray3 = plt.subplots(nrows=1,ncols=6,figsize=(20, 20))
#fig3.subplots_adjust(hspace=.5,wspace=0.4)
#bestparamM3 = []
#bestparamSigmasM3 = []
#for i in range(5):
#    axis3 = axesArray3[1, i+1]
##    a, b, c = axis3.hist(lista_v3[:,i], bins=50, density=True)
 #   bin_max = np.where(a == a.max())
 #   desv =np.std(lista_v3[:,i])
 #   mB =np.mean(lista_v3[:,i])
 #   bestparamSigmasM3.append(desv)
 #   bestparamM3.append(mB)
 #   axis3.set_title(r"Distribucion del M3 $\beta_{:.0f}$. Con un valor medio de {:.2f} $\pm$ {:.2f}".format(float(i) , float(mB), float(desv)))
 #   axis3.set_xlabel(r"$\beta_{:.0f}$".format(float(i)))#

n=np.size(Y)


#loglikeM1, loglikeM2, loglikeM3  = loglikelihood(Y,X,bestparamM1,bestparamM2,bestparamM3,Sigmas)

#loglikeM1 = (-loglikeM1 + 3/2*np.log(n))
#loglikeM2 = (-loglikeM2 + 3/2*np.log(n))
#loglikeM3 = (-loglikeM3 + 3/2*np.log(n))

#axis1 = axesArray1[1, 4]
#axis1.scatter(X,Y)
#axis1.errorbar(X, Y, yerr=Sigmas, xerr=Sigmas)
#axis1.plot(X,modelo1(bestparamM1,X))
#axis1.set_title(r"BIC ={:.0f}".format(float(loglikeM1)))
#fig1.savefig('ModeloA.png')
#axis12= axesArray1[1, 4]
#axis12.scatter(X,Y)
#axis12.errorbar(X, Y, yerr=Sigmas, xerr=Sigmas)
#axis12.plot(X,modelo2(bestparamM2,X))
#axis12.set_title(r"BIC ={:.0f}".format(float(loglikeM2)))
#fig2.savefig('ModeloB.png')
#axis3 = axesArray3[1, 6]
#axis3.scatter(X,Y)
##axis3.errorbar(X, Y, yerr=Sigmas, xerr=Sigmas)
#axis3.plot(X,modelo3(bestparamM3,X))
#axis3.set_title(r"BIC ={:.0f}".format(float(loglikeM3)))
#fig3.savefig('ModeloC.png')
