import matplotlib.pyplot as plt 
import numpy as np
import json
import os
import glob
class Analysis: 
    def __init__(self):
        oracle = os.path.expanduser("~/investigaton-NLP/investigaton-NLP-main/data/longmemeval/longmemeval_oracle.json")
    
    def make_bar_chart_wrong_answers_by_types(self):
        
        INPUT_DIR = "~/investigaton-NLP/investigaton-NLP-main/data/results/prueba8ParesConPlanINVESTIGATHONNOHELDOUT/*.json"
        INPUT_DIR = os.path.expanduser(INPUT_DIR)   

        ORDER_TYPES = ["multi-session", "single-session-preference", "temporal-reasoning", "abstention", "knowledge-update", "single-session-user", "single-session-assistant"]
        stats = {t: {'total': 0, 'correct': 0} for t in ORDER_TYPES}
        
        
        correct_count_by_type = {}
        total_count_by_type = {}
        
        for filepath in glob.glob(INPUT_DIR):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                # Lógica de tipo (mantenida de tu código)
                q_id = data.get("question_id", "")
                q_abs = data.get("original_non_abstention_question", "nosoy")
                q_type = "abstention" if q_abs != "nosoy" else data.get("question_type")
                
                # Solo sumamos si el tipo está en nuestra lista de interés
                if q_type in stats:
                    stats[q_type]['total'] += 1
                    if data.get("answer_is_correct"):
                        stats[q_type]['correct'] += 1
            except Exception as e:
                print(f"Error en {filepath}: {e}")

        # Gráfico

        names = []
        percentages = []
        labels = []

        for t in ORDER_TYPES:
            curr = stats[t]
            total = curr['total']
            correct = curr['correct']
            
            pct = (correct / total * 100) if total > 0 else 0
            formatted_name = t.replace("-", '\n')
            names.append(formatted_name)
            percentages.append(pct)
            labels.append(f"{pct:.1f}%\n({correct}/{total})")

        # 4. Gráfico Simplificado
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, percentages, color='steelblue')

        # Etiquetas sobre las barras
        for bar, label in zip(bars, labels):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                     label, ha='center', va='bottom', fontsize=10)
        
        plt.ylim(0, 110) # Margen extra para etiquetas
        plt.title("Aciertos por Tipo de Pregunta (Orden Fijo)")
        plt.ylabel("% Aciertos")
        plt.tight_layout()
        plt.savefig("prueba8ParesConPlanINVESTIGATHONNOHELDOUT.png")
        
    def make_bar_chart_total(self):
        
        INPUT_DIR = "~/investigaton-NLP/investigaton-NLP-main/data/results/prueba500conPLAN/*.json"
        INPUT_DIR = os.path.expanduser(INPUT_DIR)   
        cantidadtrues = 0
        totales = 0

        for filepath in glob.glob(INPUT_DIR):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    totales += 1
                    if data.get("answer_is_correct"):
                        cantidadtrues += 1
                        
            except Exception as e:
                print(f"Error en {filepath}: {e}")

        print(f"TRUES: {cantidadtrues}")
        print(f"TOTALES: {totales}")