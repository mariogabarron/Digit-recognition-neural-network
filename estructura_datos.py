"""
estructura_datos.py

Estructura de datos desde cero de una red neuronal de 4 capas
para la detección numérica en una malla de 25x25 píxeles :)

Autor: Mario Gabarrón Espín
"""
import numpy as np

NUMERO_CAPAS = 4

# Defino la neurona con su índice entre las n_L que hay en la capa L, la capa L donde está y su sesgo.
class Neurona:

    def __init__(self, indice , sesgo, z):
        self.indice = indice
        # self.capa = capa                      (ya lo guardará la matriz)
        self.activacion = 0                     # CORRECCIÓN de chati, mejor ponemos a 0, no hace falta asignar aleatoriedad al estado inicial
        self.sesgo = sesgo
        self.z = z

    #def metodo(self):
    #    print(self.atributo1)

# Defino las matrices de pesos (habrá 3, para cada zona entre capas) de forma aleatoria
# Después en la parte de aprendizaje estos pesos irán cambiando con el descenso del gradiente :)    

W1 = np.random.randn(625,120)*0.05
W2 = np.random.randn(120,60)*0.05
W3 = np.random.randn(60,10)*0.05

matricesPesos = [W1,W2,W3]

# Defino la función sigmoide, que tomaremos la clásica (definida de R a [0,1])
# (Hay muchas más eficientes pero uso la más famosa)

def sigmoide(x):
    return 1/(1+np.exp(-x))