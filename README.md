# Red neuronal desde cero

Implementación desde cero de una red neuronal artificial para reconocer dígitos dibujados en una malla de 25x25 píxeles.

El objetivo principal del proyecto es entender matemáticamente y programar manualmente los elementos básicos de una red neuronal:

- estructura de neuronas,
- matrices de pesos,
- función sigmoide,
- propagación hacia delante,
- función de coste,
- retropropagación,
- descenso del gradiente,
- entrenamiento,
- predicción,
- interfaz gráfica para dibujar y probar números.

La red utiliza la arquitectura:

```text
625 -> 120 -> 60 -> 10
```

donde la entrada corresponde a los 625 píxeles de una imagen de 25x25, y la salida contiene 10 neuronas asociadas a los dígitos del 0 al 9.

## Archivos principales

```text
estructura_datos.py
red_neuronal.py
interfaz.py
entrenamiento_25x25_correcto.txt
Memoria/MemoriaRedNeuronal.pdf
```

## Ejecución

Crea un entorno virtual e instala las dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Después ejecuta:

```bash
python interfaz.py
```

## Dependencias

El proyecto usa principalmente:

- Python 3
- NumPy
- Tkinter

En algunas instalaciones de macOS con Homebrew puede hacer falta instalar Tkinter aparte, por ejemplo:

```bash
brew install python-tk
```

## Nota

Este proyecto se ha desarrollado con intención educativa. No utiliza librerías de machine learning como TensorFlow, Keras o PyTorch, precisamente para implementar manualmente el funcionamiento interno de la red.
