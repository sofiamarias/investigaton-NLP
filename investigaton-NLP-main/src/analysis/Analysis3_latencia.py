import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import json
import os

class Analysis3: 
    def __init__(self):
        results_dir = (f"/home/ubuntu/investigaton-NLP/investigaton-NLP-main/data/results/500k10_judge_midiendoSoloLatencia")
        json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
        print(len(json_files))
        results = []
        for filename in json_files:
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                results.append(data)

        self.df = pd.DataFrame(results)

            
    def make_latency_chart(self):
        metrics = self.df.groupby("question_type").agg(
        tiempo_en_segs=("time taken to answer", "mean")  # Calcula el promedio (0.0 a 1.0)
        ).sort_values(by="tiempo_en_segs", ascending=True).reset_index()

        #momento grafico
        plt.style.use('seaborn-v0_8-whitegrid') # O 'ggplot' si prefieres
        fig, ax = plt.subplots(figsize=(10, 7))

        # barras
        barras = ax.barh(metrics['question_type'] , metrics['tiempo_en_segs'], color='#4c72b0', zorder=3)
        plt.subplots_adjust(left=0.2)  # esto porque me quedan cortadas
        ax.set_xlim(0, 15) 
        ax.xaxis.set_major_formatter(
                    mtick.FuncFormatter(lambda x, pos: f"{x:.0f} segs"))


        ax.set_title(' Tiempo promedio en segundos por cada tipo de Pregunta', fontsize=14, pad=15)

        for barra in barras:
            width = barra.get_width()
            # El 3f para que agarre algunos decimales (se ve mas pro)
            label_text = f'{width:.3f} segs ' 
            
            ax.text(width + 0.01,       
                    barra.get_y() + barra.get_height()/2,
                    label_text, 
                    ha='left', va='center', fontsize=10, fontweight='bold', color='black')

        plt.savefig("muestra500-latenciaPromedio.png")
        plt.show()

        return