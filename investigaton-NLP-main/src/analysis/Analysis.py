import matplotlib.pyplot as plt 
import numpy as np
import json
import os
import glob
class Analysis: 
    def __init__(self):
        pass
    
    def make_bar_chart_wrong_answers_by_types(self):
        
        INPUT_DIR = "~/investigaton-NLP/investigaton-NLP-main/data/results/longmemeval/short/versionpromptsofi/*.json"
        INPUT_DIR = os.path.expanduser(INPUT_DIR)   
        question_types = []
        answers = []
        wrong_count_by_type = {}
        total_count_by_type = {}
        p = glob.glob(INPUT_DIR)
        for filepath in glob.glob(INPUT_DIR):
            with open(filepath, "r") as f:
                data = json.load(f)
        
            qname = data["question_id"]
            if qname.endswith("_abs"):
                qtype = "abstention"
            else:
                qtype = data.get("question_type")
            qanswer = data["answer_is_correct"]
            total_count_by_type[qtype] = total_count_by_type.get(qtype, 0) + 1
            if(not(qanswer)):
                wrong_count_by_type[qtype] = wrong_count_by_type.get(qtype, 0) + 1

        # Gráfico
        fractions = {t: f"{wrong_count_by_type.get(t, 0)} / {total_count_by_type[t]}"
             for t in total_count_by_type}
        types = list(wrong_count_by_type.keys())
        #counts = list(wrong_count_by_type.values())

        plt.figure(figsize=(10,5))
        
        counts = []
        for i, t in enumerate(types):
            wrong = wrong_count_by_type.get(t, 0)
            total = total_count_by_type[t]
            ratio = round(wrong * 100 / total, 1)
            counts.append(ratio)
            plt.text(i, ratio + 0.5 , f"{ratio}% {total}", ha='center')
        bars = plt.bar(types, counts)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("abc.png")
        plt.show()

