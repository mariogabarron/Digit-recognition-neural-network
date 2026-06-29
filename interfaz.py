"""
interfaz.py

Interfaz gráfica para dibujar números en una malla 25x25,
guardar ejemplos propios, visualizar ejemplos de entrenamiento
y probar la red neuronal definida en red_neuronal.py.

IMPORTANTE:
Realizada por ChatGPT, pero adaptada. Este proyecto al final
tenía como ejercicio principal realizar la red neuronal desde cero, y la interfaz es un añadido
para poder probarla. La interfaz no es el objetivo principal del proyecto, pero sí es útil
para probar la red neuronal y ver cómo funciona.

Autor: Mario Gabarrón Espín
"""

import tkinter as tk
from tkinter import messagebox, simpledialog
import json
import os
import numpy as np

import red_neuronal


GRID_SIZE = 25
PIXEL_SIZE = 20
PREVIEW_PIXEL_SIZE = 6

ARCHIVO_BASE = "entrenamiento_25x25_correcto.txt"
ARCHIVO_MIS_EJEMPLOS = "mis_ejemplos_25x25.txt"

ANIMAR_CADA_N_IMAGENES = 5


def one_hot(numero):
    return [1 if i == numero else 0 for i in range(10)]


class PaintApp:

    def __init__(self):

        self.root = tk.Tk()
        self.root.title("Red neuronal - Dibujar número")

        self.matrix = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        self.entrenada = False
        self.entrenando = False

        self.canvas = tk.Canvas(
            self.root,
            width=GRID_SIZE * PIXEL_SIZE,
            height=GRID_SIZE * PIXEL_SIZE,
            bg="white"
        )

        self.canvas.pack(padx=10, pady=10)

        self.draw_grid()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.canvas.bind("<B3-Motion>", self.erase)
        self.canvas.bind("<Button-3>", self.erase)

        botones = tk.Frame(self.root)
        botones.pack(pady=10)

        self.boton_entrenar = tk.Button(
            botones,
            text="Entrenar",
            command=self.entrenar_modelo
        )
        self.boton_entrenar.pack(side=tk.LEFT, padx=5)

        tk.Button(
            botones,
            text="Reconocer",
            command=self.reconocer
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            botones,
            text="Guardar ejemplo",
            command=self.guardar_ejemplo
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            botones,
            text="Limpiar",
            command=self.clear
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            botones,
            text="Imprimir matriz",
            command=self.print_matrix
        ).pack(side=tk.LEFT, padx=5)

        self.resultado = tk.Label(
            self.root,
            text="Dibuja un número y pulsa Reconocer",
            font=("Arial", 18)
        )
        self.resultado.pack(pady=10)

        self.preview_label = tk.Label(
            self.root,
            text="Imagen de entrenamiento actual:",
            font=("Arial", 13)
        )
        self.preview_label.pack(pady=(5, 0))

        self.preview_canvas = tk.Canvas(
            self.root,
            width=GRID_SIZE * PREVIEW_PIXEL_SIZE,
            height=GRID_SIZE * PREVIEW_PIXEL_SIZE,
            bg="white"
        )
        self.preview_canvas.pack(pady=5)

        self.preview_rects = []

        for i in range(GRID_SIZE):
            fila = []
            for j in range(GRID_SIZE):

                x0 = j * PREVIEW_PIXEL_SIZE
                y0 = i * PREVIEW_PIXEL_SIZE
                x1 = x0 + PREVIEW_PIXEL_SIZE
                y1 = y0 + PREVIEW_PIXEL_SIZE

                rect = self.preview_canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill="white",
                    outline="gray"
                )

                fila.append(rect)

            self.preview_rects.append(fila)

        self.preview_info = tk.Label(
            self.root,
            text="Todavía no se está entrenando.",
            font=("Arial", 12)
        )
        self.preview_info.pack(pady=(0, 8))

        self.activaciones_texto = tk.Text(
            self.root,
            width=55,
            height=12,
            font=("Courier", 12)
        )
        self.activaciones_texto.pack(padx=10, pady=10)

        self.activaciones_texto.insert(
            tk.END,
            "Aquí aparecerán las activaciones de las neuronas 0-9.\n"
        )

    def draw_grid(self):

        self.canvas.delete("grid")

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):

                x0 = j * PIXEL_SIZE
                y0 = i * PIXEL_SIZE
                x1 = x0 + PIXEL_SIZE
                y1 = y0 + PIXEL_SIZE

                color = "black" if self.matrix[i][j] else "white"

                self.canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill=color,
                    outline="gray",
                    tags="grid"
                )

    def paint(self, event):

        col = event.x // PIXEL_SIZE
        row = event.y // PIXEL_SIZE

        self.paint_square(row, col, value=1)

    def erase(self, event):

        col = event.x // PIXEL_SIZE
        row = event.y // PIXEL_SIZE

        self.paint_square(row, col, value=0)

    def paint_square(self, row, col, value):

        for dr in range(-1, 2):
            for dc in range(-1, 2):

                r = row + dr
                c = col + dc

                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:

                    self.matrix[r][c] = value

                    x0 = c * PIXEL_SIZE
                    y0 = r * PIXEL_SIZE

                    color = "black" if value == 1 else "white"

                    self.canvas.create_rectangle(
                        x0,
                        y0,
                        x0 + PIXEL_SIZE,
                        y0 + PIXEL_SIZE,
                        fill=color,
                        outline="gray",
                        tags="grid"
                    )

    def clear(self):

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.matrix[i][j] = 0

        self.draw_grid()

        self.resultado.config(text="Dibuja un número y pulsa Reconocer")
        self.activaciones_texto.delete("1.0", tk.END)
        self.activaciones_texto.insert(
            tk.END,
            "Aquí aparecerán las activaciones de las neuronas 0-9.\n"
        )

    def get_matrix(self):
        return self.matrix

    def print_matrix(self):

        for fila in self.matrix:
            print(fila)

        print()

    def cargar_datos_entrenamiento(self):

        with open(ARCHIVO_BASE, "r", encoding="utf-8") as f:
            datos = json.load(f)

        imagenes = datos["imagenes"]
        etiquetas = datos["etiquetas"]

        num_base = len(imagenes)
        num_propios = 0

        if os.path.exists(ARCHIVO_MIS_EJEMPLOS):
            with open(ARCHIVO_MIS_EJEMPLOS, "r", encoding="utf-8") as f:
                datos_propios = json.load(f)

            imagenes = imagenes + datos_propios["imagenes"]
            etiquetas = etiquetas + datos_propios["etiquetas"]
            num_propios = len(datos_propios["imagenes"])

        return imagenes, etiquetas, num_base, num_propios

    def mostrar_imagen_entrenamiento(self, imagen, etiqueta, epoch, posicion, total):

        if posicion % ANIMAR_CADA_N_IMAGENES != 0:
            return

        numero = int(np.argmax(etiqueta))

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):

                color = "black" if imagen[i][j] == 1 else "white"

                self.preview_canvas.itemconfig(
                    self.preview_rects[i][j],
                    fill=color
                )

        self.preview_info.config(
            text="Epoch " + str(epoch) +
                 " | Imagen " + str(posicion + 1) + "/" + str(total) +
                 " | Etiqueta real: " + str(numero)
        )

        self.root.update_idletasks()
        self.root.update()

    def guardar_ejemplo(self):

        numero = simpledialog.askinteger(
            "Guardar ejemplo",
            "¿Qué número has dibujado? Escribe un número del 0 al 9:",
            minvalue=0,
            maxvalue=9
        )

        if numero is None:
            return

        imagen = [fila[:] for fila in self.matrix]
        etiqueta = one_hot(numero)

        if os.path.exists(ARCHIVO_MIS_EJEMPLOS):
            with open(ARCHIVO_MIS_EJEMPLOS, "r", encoding="utf-8") as f:
                datos = json.load(f)
        else:
            datos = {
                "descripcion": "Ejemplos propios dibujados desde la interfaz.",
                "grid_size": GRID_SIZE,
                "num_samples": 0,
                "labels": [],
                "imagenes": [],
                "etiquetas": []
            }

        datos["imagenes"].append(imagen)
        datos["etiquetas"].append(etiqueta)
        datos["labels"].append(numero)
        datos["num_samples"] = len(datos["imagenes"])

        with open(ARCHIVO_MIS_EJEMPLOS, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False)

        self.resultado.config(
            text="Ejemplo guardado como número " + str(numero) +
                 " ✅ Total propios: " + str(datos["num_samples"])
        )

        messagebox.showinfo(
            "Ejemplo guardado",
            "Guardado como número " + str(numero) + ".\n\n"
            "Archivo: " + ARCHIVO_MIS_EJEMPLOS + "\n"
            "Total de ejemplos propios: " + str(datos["num_samples"])
        )

    def entrenar_modelo(self):

        if self.entrenando:
            messagebox.showinfo(
                "Entrenamiento en curso",
                "La red ya se está entrenando."
            )
            return

        try:
            imagenes, etiquetas, num_base, num_propios = self.cargar_datos_entrenamiento()

            self.entrenando = True
            self.boton_entrenar.config(state=tk.DISABLED)

            self.resultado.config(text="Entrenando...")
            self.activaciones_texto.delete("1.0", tk.END)
            self.activaciones_texto.insert(
                tk.END,
                "Entrenamiento iniciado.\n\n"
                "Ejemplos base: " + str(num_base) + "\n"
                "Ejemplos propios: " + str(num_propios) + "\n"
                "Total: " + str(len(imagenes)) + "\n\n"
            )

            red_neuronal.entrenar(
                imagenes,
                etiquetas,
                epochs=9,
                callback_imagen=self.mostrar_imagen_entrenamiento
            )

            precision = red_neuronal.evaluar(imagenes, etiquetas)

            self.entrenada = True
            self.entrenando = False
            self.boton_entrenar.config(state=tk.NORMAL)

            self.resultado.config(
                text="Entrenamiento terminado ✅ Precisión: " +
                     str(round(precision * 100, 2)) + "%"
            )

            self.activaciones_texto.insert(
                tk.END,
                "Entrenamiento terminado.\n"
                "Precisión final: " + str(round(precision * 100, 2)) + "%\n"
            )

            self.preview_info.config(text="Entrenamiento terminado.")
            
        except FileNotFoundError:
            self.entrenando = False
            self.boton_entrenar.config(state=tk.NORMAL)

            messagebox.showerror(
                "Archivo no encontrado",
                "No encuentro " + ARCHIVO_BASE + ".\n\n"
                "Pon ese archivo en la misma carpeta que interfaz.py."
            )

            self.resultado.config(text="Error: falta el archivo de entrenamiento")

        except Exception as e:
            self.entrenando = False
            self.boton_entrenar.config(state=tk.NORMAL)

            messagebox.showerror(
                "Error durante el entrenamiento",
                str(e)
            )

            self.resultado.config(text="Error durante el entrenamiento")

    def reconocer(self):

        try:
            numero = int(red_neuronal.predecir(self.matrix))

            red_neuronal.initNeuronalNetwork(self.matrix)
            salida = red_neuronal.forwarding()
            activaciones = [neurona.activacion for neurona in salida]

            self.resultado.config(
                text="La red cree que es un: " + str(numero)
            )

            self.activaciones_texto.delete("1.0", tk.END)

            for i, a in enumerate(activaciones):
                self.activaciones_texto.insert(
                    tk.END,
                    "Neurona " + str(i) + ": " + str(round(a, 6)) + "\n"
                )

        except Exception as e:
            messagebox.showerror(
                "Error al reconocer",
                str(e)
            )

    def run(self):
        self.root.mainloop()

# Función principal para ejecutar la interfaz

if __name__ == "__main__":

    app = PaintApp()
    app.run()
