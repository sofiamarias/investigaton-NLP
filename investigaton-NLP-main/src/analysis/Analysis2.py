import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import json
import os

class Analysis: 
    def __init__(self, set, type, sample):
        results_dir = (f"/home/ubuntu/investigaton-NLP/investigaton-NLP-main/data/results/{set}/{type}/{sample}")
        json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]

        results = []
        for filename in json_files:
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                results.append(data)

        self.df = pd.DataFrame(results)

            
    def make_bar_chart_right_answers_by_types(self):
        # 1. PREPARAR LOS DATOS (Agrupación)
        # Asumimos que 'df' ya está cargado con tu código anterior
        metrics = self.df.groupby("question_type").agg(
        accuracy=("answer_is_correct", "mean"),  # Calcula el promedio (0.0 a 1.0)
        count=("answer_is_correct", "count")     # Cuenta cuántos casos hay
        ).sort_values(by="accuracy", ascending=True).reset_index()

        # 2. CONFIGURAR EL GRÁFICO
        # Usamos un estilo limpio
        plt.style.use('seaborn-v0_8-whitegrid') # O 'ggplot' si prefieres
        fig, ax = plt.subplots(figsize=(10, 6))

        # 3. DIBUJAR LAS BARRAS
        # 'barh' crea barras horizontales.
        # Zorder=3 hace que las barras queden encima de la rejilla de fondo.
        barras = ax.barh(metrics['question_type'], metrics['accuracy'], color='#4c72b0', zorder=3)

        # 4. FORZAR EL EJE AL 100% (Tu requerimiento principal)
        ax.set_xlim(0, 1.05) # Ponemos 1.05 para dar un poquito de aire extra a la derecha

        # 5. FORMATEAR EL EJE X COMO PORCENTAJE
        # Esto cambia los números "0.2, 0.4" por "20%, 40%"
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        # 6. ETIQUETAS Y TÍTULOS
        ax.set_title('Precisión (Accuracy) por Tipo de Pregunta', fontsize=14, pad=15)
        ax.set_xlabel('Porcentaje de Aciertos', fontsize=12)
        ax.set_ylabel('') # No hace falta etiqueta en Y porque los nombres ya explican

        # 7. AGREGAR LOS VALORES AL LADO DE CADA BARRA
        # Esto le da un toque muy profesional
        for barra in barras:
            width = barra.get_width()
            # Escribimos el porcentaje y la cantidad (n)
            label_text = f'{width:.1%} ' 
            
            ax.text(width + 0.01,       # Posición X (un poco a la derecha del final de la barra)
                    barra.get_y() + barra.get_height()/2, # Posición Y (centro de la barra)
                    label_text, 
                    ha='left', va='center', fontsize=10, fontweight='bold', color='black')

        # 8. MOSTRAR
        plt.tight_layout()
        plt.savefig("muestra500-cuantorespondiobien.png")
        plt.show()

        return