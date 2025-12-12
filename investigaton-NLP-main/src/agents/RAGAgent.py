import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.datasets.LongMemEvalDataset import LongMemEvalInstance
import json
class RAGAgent:
    def __init__(self, model, semantic_retriever_agent, contextualizer_agent):
        self.model = model
        self.semantic_retriever_agent = semantic_retriever_agent
        self.contextualizer_agent = contextualizer_agent
        self.cantidaddetrues = 0
        self.sessions_id_used_by_question = {}
        self.cross_encoder_scores_used_by_question = {}
    
    def get_sessions_used_by(self, question_id):
        return self.sessions_id_used_by_question.get(question_id, [])
    def get_cross_encoders_used_by(self, question_id):
        return self.cross_encoder_scores_used_by_question.get(question_id, [])

    def answer(self, instance: LongMemEvalInstance):
        #Conseguimos los documentos más relevantes
        topk_relevant_documents, topk_relevant_scores = self.semantic_retriever_agent.retrieve_most_relevant_messages(instance)
        if not topk_relevant_documents:
            return "I do not have enough information"
        
        # 2. Formateo de Evidencia (XML estricto para Gemma 3)
        evidence_block = ""
        for doc, score in zip(topk_relevant_documents, topk_relevant_scores):
            # Sanitización de caracteres reservados
            safe_pair = doc['full_pair_text'].replace("<", "&lt;").replace(">", "&gt;")
            evidence_block += f"""
            <log id="{doc['id']}" date="{doc['timestamp']}">
            {safe_pair}
            </log>
            """
            #Guardamos las sesiones usadas
            self.sessions_id_used_by_question.setdefault(instance.question_id, []).append(f"{doc['session_id']}: {doc['id']}")      
            self.cross_encoder_scores_used_by_question.setdefault(instance.question_id, []).append(f"{doc['id']}: {score}")      
        raw_instruction = self.answer_format(instance.question)
        format_instruction = raw_instruction.replace("Plan:", "").replace("Format Plan:", "").replace("PLAN:", "").strip()            
        prompt = f"""
        You are an expert Personal Memory Assistant. Your goal is to answer the user's question by analyzing their past conversation logs.

        ### RULES OF REASONING
        1. **Knowledge Update (State Tracking):** Old information can be overwritten by newer logs. If the user had a bike in 2020 but sold it in 2022, they have 0 bikes. Always check the dates.
        2. **Implicit Preferences (Single-Session Preference):** If the user asks for a recommendation, look at their past habits or owned devices in the logs. (e.g., If they own a Sony TV, recommend compatible Sony speakers).
        
        ### REAL DATA (Analyze this)
        <memory_database>
        {evidence_block}
        </memory_database>

        ### CURRENT QUESTION
        **Question:** {instance.question}
        **Plan:**
        {format_instruction}
        
        **Answer:**
        """
 
        messages = [{"role": "user", "content": prompt}]
        answer = self.model.reply(messages, temperature=0.0)
        return answer
    def answer_format(self, question):
        prompt = f"""
        ### TASK
        Classify the Question and provide a concise execution plan. Output ONLY the plan.

        ### EXAMPLES

        Q: "How many days did I camp?"
        Plan: 1. Identify unique trips by location. 2. Ignore Assistant repetitions. 3. List trips with durations. 4. Answer the sum of the total days.

        Q: "How old was I when Alex was born?"
        Plan: 1. Identify the user age. 2. Identify the Alex age. 3. Answer the difference between the ages.
        
        Q: "How long before Google did I work?"
        Plan: 1. Check if the EVIDENCE explicitly confirms USER worked at Google. 2. If no, state "I do not have enough information". 3. If yes, calculate time difference.


        Q: "Why is my bike faster?"
        Plan: Scan previous EVIDENCE for maintenance actions (e.g., "changed chain") that explain the improvement.

        Q: "Tips for Tokyo?"
        Plan: Identify tools/apps mentioned in EVIDENCE (e.g., "Suica"). Create tips using ONLY those resources.

        Q: "Do I have 30-gallon tanks?"
        Plan: Check EVIDENCE for "30-gallon". If only "18" or "20" are found, state "Information not available".

        Q: Which happened first: finishing The Office or attending the charity dinner?...
        Plan: 1. Find Finishing The Office date. 2. Find the charity dinner date. 3. Answer the most recent event.
        
        Q: "What was the name of that hostel near the Red Light District that you recommended last time?"
        Plan: 1. Search EVIDENCE for previous recommendatios from YOU to the USER near the Red Light District. 2. Extract the hostel name.

        ### INPUT
        Q: "{question}"

        ### PLAN:
        """
       
        response = [{"role": "user", "content": prompt}]
        answer = self.model.reply(response, temperature = 0.0)
        return answer
    