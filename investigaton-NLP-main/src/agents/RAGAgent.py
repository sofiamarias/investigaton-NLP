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

    
    def get_sessions_used_by(self, question_id):
        return self.sessions_id_used_by_question.get(question_id, [])

    def evidence_is_enough(self, question, evidence):
        prompt = f"""
        You are a chatbot agent and need to decide if you have enough information in order to answer the question. Analize all the information before answering. Be optimistic while searching, realistic to answer.
        Justify your answer.
        Question: {question}
        Evidence to evaluate: {evidence}    
        """
        messages = [{"role": "user", "content": prompt}]
        answer = self.model.reply(messages, temperature = 0.0)
        return answer

    def get_relevant_info_by_query(self, pair, question):
        prompt = f"""
        You are a Data Extractor Engine. Given a question and a pair of messages between an AI agent and a user, answer a bullet points list with the relevant data from the pair to answer the question. 

        #Example
        Question: How many classes of History did I take this week?
        User: I've taken 2 History classes on Tuesday and one on Monday. How many more should I take?
        Assistant: You should take two more classes to be ready for the exam.
        Output: 
        * The user has taken two History classes on Tuesday 
        * The user has taken one History class on Monday.
        
        ## QUESTION
        {question}

        ## PAIR
        {pair}

        ##ANSWER
        """
        messages = [{"role": "user", "content": prompt}]
        answer = self.model.reply(messages, temperature = 0.0)
        return answer

    def answer(self, instance: LongMemEvalInstance):
        #Conseguimos los documentos más relevantes
        topk_relevant_documents = self.semantic_retriever_agent.retrieve_most_relevant_messages(instance)
        if not topk_relevant_documents:
            return "I do not have enough information"
        
        evidence = []
        for document in topk_relevant_documents:
            entry = {
                "date": document['timestamp'],
                #Resumimos el par (user, ia)
                "evidence": self.get_relevant_info_by_query(document['full_pair_text'], f"[{instance.question_date}]: {instance.question}")
            }
            
            evidence.append(json.dumps(entry, ensure_ascii=False))
            #Guardamos las sesiones usadas
            self.sessions_id_used_by_question.setdefault(instance.question_id, []).append(f"{document['session_id']}: {document['id']}")      

        evidence_text = "\n".join(evidence)
        print(evidence_text)
        #Vemos si la evidencia es suficiente para responder la pregunta (bool)
    
        raw_instruction = self.answer_format(instance.question)
        format_instruction = raw_instruction.replace("Plan:", "").replace("Format Plan:", "").replace("PLAN:", "").strip()            
        print(format_instruction)
        prompt = f"""
        ### ROLE
        You are a precise Research Assistant. You answer questions based ONLY on the provided memory logs.

        ### MEMORY LOGS (Chronological)
        {evidence_text}

         ### TASK
        Question: [{instance.question_date}] {instance.question}
        
        ### EXECUTION PLAN
        {format_instruction}

        ### RULES
        1. **Conflicts:** In case of conflicts or contradictions, recent logs SUPERSEDE older logs.       
        2. **User Preferences (Implicit Constraints):** If the user asks a general question (e.g., "tips for video editing"), check the logs for specific tools, software, locations, or habits they already use.
        ### ANSWER: 
        """
        messages = [{"role": "user", "content": prompt}]
        answer = self.model.reply(messages, temperature = 0.0)
        return answer
    
    def answer_format(self, question):
        prompt = f"""
        ### TASK
        Classify the Question and provide a concise execution plan. Output ONLY the plan.

        ### EXAMPLES

        Q: "How many days did I camp?"
        Plan: 1. Identify unique trips by location. 2. Ignore Assistant repetitions. 3. List trips with durations. 4. Sum the total days.

        Q: "How long before Google did I work?"
        Plan: 1. Check if the EVIDENCE explicitly confirms USER worked at Google. 2. If no, state "premise not found". 3. If yes, calculate time difference.

        Q: "Why is my bike faster?"
        Plan: Scan previous EVIDENCE for maintenance actions (e.g., "changed chain") that explain the improvement.

        Q: "Tips for Tokyo?"
        Plan: Identify tools/apps mentioned in EVIDENCE (e.g., "Suica"). Create tips using ONLY those resources.

        Q: "Do I have 30-gallon tanks?"
        Plan: Check EVIDENCE for "30-gallon". If only "18" or "20" are found, state "Information not available".

        Q: "What was the name of that hostel near the Red Light District that you recommended last time?"
        Plan: 1. Search EVIDENCE for previous recommendatios from YOU to the USER near the Red Light District. 2. Extract the hostel name.

        ### INPUT
        Q: "{question}"

        ### PLAN:
        """
       
        response = [{"role": "user", "content": prompt}]
        answer = self.model.reply(response, temperature = 0.0)
        return answer
    