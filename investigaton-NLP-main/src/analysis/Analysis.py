import matplotlib.pyplot as plt 
import numpy as np
import json
import os
import glob
class Analysis: 
    def __init__(self):
        pass
    
    def make_bar_chart_wrong_answers_by_types(self):
        INPUT_DIR = "~/investigaton-NLP/investigaton-NLP-main/data/results/longmemeval/oracle/k10/*.json"
        INPUT_DIR = os.path.expanduser(INPUT_DIR)   
        question_types = []
        answers = []
        wrong_count_by_type = {}
        p = glob.glob(INPUT_DIR)
        for filepath in glob.glob(INPUT_DIR):
            with open(filepath, "r") as f:
                data = json.load(f)

            qtype = data["question_type"]
            qanswer = data["answer_is_correct"]
            
            if(not(qanswer)):
                wrong_count_by_type[qtype] = wrong_count_by_type.get(qtype, 0) + 1

        # Gráfico
        
        plt.bar(wrong_count_by_type.keys(), wrong_count_by_type.values())
        plt.savefig("a.png")
        plt.show()

