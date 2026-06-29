"""
red_neuronal.py

Implementación desde cero de una red neuronal de 4 capas
para la detección numérica en una malla de 25x25 píxeles :)

Autor: Mario Gabarrón Espín
"""
import numpy as np
from estructura_datos import Neurona, sigmoide, NUMERO_CAPAS, matricesPesos

TAMAÑOS_CAPAS = [625,120,60,10]

# Creamos las neuronas de nuestra red.

capas = []

for l in range(0, NUMERO_CAPAS):

    neuronas = []

    for k in range(0,TAMAÑOS_CAPAS[l]):
            neuronas.append(Neurona(k,0,0))    

    capas.append(neuronas)

# Inicializar la entrada de la red neuronal dada una matriz de valores entre 0 y 1 en R.
# Entrada: matriz de valores reales en [0,1]

def initNeuronalNetwork(activacionesCapa0):
    vectorCapa0 = parseMatrixToVectorRn2(activacionesCapa0)
    for i in range(0,TAMAÑOS_CAPAS[0]):
        capas[0][i].activacion = vectorCapa0[i]
    

# Definamos un forwarding: calcular la activación de cada una de las neuronas.
# Entrada: 625 valores entre 0 y 1
# Salida: 10 neuronas

def forwarding():

    for l in range(1, NUMERO_CAPAS):
        for j in range(0, TAMAÑOS_CAPAS[l]):
            y = 0
            for m in range(0, TAMAÑOS_CAPAS[l-1]):
                y += capas[l-1][m].activacion*matricesPesos[l-1][m][j] # aquí m recorre todas las neuronas de la capa L-1, j recorre las neuronas de la capa L.
            z = y + capas[l][j].sesgo
            capas[l][j].z = z
            capas[l][j].activacion = sigmoide(z)

    return capas[NUMERO_CAPAS-1]

def parseMatrixToVectorRn2(matrix):
    vector = []
    for i in range(0,len(matrix)):
        for j in range(0, len(matrix)):
            vector.append(matrix[i][j])
    return vector

# Defino la función de costes.
# Entrada: vector de neuronas de la capa final y vector objetivo (e.g.: [0,0,0,1,0,0,0,0,0] representa el 3)

def coste(neuronasFinal, vectorObjetivo):
    y=0
    for i in range(0,TAMAÑOS_CAPAS[3]):
        y += np.pow(neuronasFinal[i].activacion - vectorObjetivo[i],2)
    return y/TAMAÑOS_CAPAS[3]   # Tasa de error

# Retropropagación: nos servirá para calcular el gradiente de la función C (coste)
# para así poder aplicar el método del descenso del gradiente, y hallar los mínimos locales de la función de costes,
# para conseguir que nuestra red neuronal vaya aprendiendo.

"""

ASÍ ERA COMO LO HACÍA ANTES, PERO ES INEFICIENTE YA QUE AL FINAL ACABO HACIENDO 3 FORS, Y PARA UN PROBLEMA
DE 20 CAPAS, TENDRÍA QUE HACER 20 FORS. ES MEJOR HACERLO CON MATRICES. LA MATEMÁTICA USADA SERÁ LA MISMA, PERO MÁS EFICIENTE.

def retropropagacion(vectorObjetivo):

    dW0 = np.zeros((625,120))
    dW1 = np.zeros((120,60))
    dW2 = np.zeros((60,10))

    # ========= dW2 =========

    for j in range(60):
        for k in range(10):

            x = (capas[3][k].activacion - vectorObjetivo[k]) / 5

            y = capas[3][k].activacion * (1 - capas[3][k].activacion)

            z = capas[2][j].activacion

            dW2[j][k] = x * y * z

    # ========= dW1 =========

    for j in range(120):
        for k in range(60):

            x = 0

            for r in range(10):

                x += (
                    matricesPesos[2][k][r]
                    * ((capas[3][r].activacion - vectorObjetivo[r]) / 5)
                    * capas[3][r].activacion
                    * (1 - capas[3][r].activacion)
                )

            y = capas[2][k].activacion * (1 - capas[2][k].activacion)

            z = capas[1][j].activacion

            dW1[j][k] = x * y * z

    # ========= dW0 =========

    for j in range(625):
        for k in range(120):

            x = 0

            for r in range(60):

                aux = 0

                for s in range(10):

                    aux += (
                        matricesPesos[2][r][s]
                        * ((capas[3][s].activacion - vectorObjetivo[s]) / 5)
                        * capas[3][s].activacion
                        * (1 - capas[3][s].activacion)
                    )

                x += (
                    matricesPesos[1][k][r]
                    * aux
                    * capas[2][r].activacion
                    * (1 - capas[2][r].activacion)
                )

            y = capas[1][k].activacion * (1 - capas[1][k].activacion)

            z = capas[0][j].activacion

            dW0[j][k] = x * y * z

    return [dW0, dW1, dW2]
"""

"""
    Esta solución, sin embargo, hace uso de las deltas. Sin embargo, las
    deltas están definidas como la derivada de la función de coste con respecto a la activación de cada neurona. 
    Es la misma matemática que la anterior, pero más eficiente. 
"""

def retropropagacion(vectorObjetivo):

    dW0 = np.zeros((625,120))
    dW1 = np.zeros((120,60))
    dW2 = np.zeros((60,10))

    delta3 = np.zeros(10)
    delta2 = np.zeros(60)
    delta1 = np.zeros(120)

    # ========= CAPA DE SALIDA =========

    for k in range(10):

        a = capas[3][k].activacion

        delta3[k] = ((a-vectorObjetivo[k])/5) * a * (1-a)

    for j in range(60):
        for k in range(10):

            dW2[j][k] = capas[2][j].activacion * delta3[k]

    # ========= CAPA OCULTA 2 =========

    for k in range(60):

        suma = 0

        for r in range(10):
            suma += matricesPesos[2][k][r] * delta3[r]

        a = capas[2][k].activacion

        delta2[k] = suma * a * (1-a)

    for j in range(120):
        for k in range(60):

            dW1[j][k] = capas[1][j].activacion * delta2[k]

    # ========= CAPA OCULTA 1 =========

    for k in range(120):

        suma = 0

        for r in range(60):
            suma += matricesPesos[1][k][r] * delta2[r]

        a = capas[1][k].activacion

        delta1[k] = suma * a * (1-a)

    for j in range(625):
        for k in range(120):

            dW0[j][k] = capas[0][j].activacion * delta1[k]

    return [dW0, dW1, dW2, delta1, delta2, delta3]

# Realizamos el descenso del gradiente modificando los pesos y los sesgos según el gradiente (lo obtenemos de la retropropagación)

def descensoGradiente(gradiente):
    eta = 1

    matricesPesos[0] -= eta * gradiente[0]
    matricesPesos[1] -= eta * gradiente[1]
    matricesPesos[2] -= eta * gradiente[2]

    for i in range(120):
        capas[1][i].sesgo -= eta * gradiente[3][i]

    for i in range(60):
        capas[2][i].sesgo -= eta * gradiente[4][i]

    for i in range(10):
        capas[3][i].sesgo -= eta * gradiente[5][i]

# Entrenamos nuestro modelo

def entrenar(imagenes, etiquetas, epochs, callback_imagen=None):
    for e in range(epochs):

        coste_total = 0

        # Esto es para que las imágenes se barajen en cada epoch, para que no se aprenda un patrón de ordenación de las imágenes.
        indices = np.random.permutation(len(imagenes))

        for posicion, i in enumerate(indices):

            if callback_imagen is not None:
                callback_imagen(imagenes[i], etiquetas[i], e, posicion, len(imagenes))

            initNeuronalNetwork(imagenes[i])
            salida = forwarding()

            coste_total += coste(salida, etiquetas[i])

            gradiente = retropropagacion(etiquetas[i])
            descensoGradiente(gradiente)

        precision = evaluar(imagenes, etiquetas)

        print("Epoch", e)
        print("Coste medio:", coste_total / len(imagenes))
        print("Precisión:", precision)
        print("--------------------")

        if precision == 1.0:
            print("Precisión perfecta. Paramos entrenamiento.")
            break

    return capas[NUMERO_CAPAS-1]

def predecir(imagen):

    initNeuronalNetwork(imagen)
    salida = forwarding()
    salida = [n.activacion for n in salida]
    return np.argmax(salida) # Devuelve el índice de la neurona con mayor activación.


def evaluar(imagenes, etiquetas):

    aciertos = 0

    for i in range(len(imagenes)):
        prediccion = predecir(imagenes[i])
        real = np.argmax(etiquetas[i])
        if prediccion == real:
            aciertos += 1

    precision = aciertos / len(imagenes) # Tasa de aciertos
    print("Aciertos:", aciertos, "/", len(imagenes))
    print("Precisión:", precision)

    return precision



