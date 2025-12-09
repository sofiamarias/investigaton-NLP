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

        self.sessions_id_used_by_question = {}

    
    def get_sessions_used_by(self, question_id):
        return self.sessions_id_used_by_question.get(question_id, [])

    def answer(self, instance: LongMemEvalInstance):
        #Conseguimos los documentos más relevantes
        topk_relevant_documents = self.semantic_retriever_agent.retrieve_most_relevant_messages(instance)
        if not topk_relevant_documents:
            return "I do not have enough information"
        
        evidence = []
        for document in topk_relevant_documents:
            evidence_dict = {
                "timestamp": document["timestamp"],
                "dialogue": document["full_pair_text"]
            }
            evidence_str = json.dumps(evidence_dict)
            evidence.append(evidence_str)
            #Guardamos las sesiones usadas
            self.sessions_id_used_by_question.setdefault(instance.question_id, []).append(f"{document['session_id']}: {document['id']}")      

          
        #Todavia no funciona
        #list_contextualized_topk_relevant_messages = self.contextualizer_agent.retrieve_contextualized_messages(topk_relevant_messages, topk_relevant_sessions)
        #contextualized_topk_relevant_messages = "\n\n".join(list_contextualized_topk_relevant_messages)
        
        format_instruction = self.answer_format(instance.question)
        
        prompt = rag_prompt = f"""
        ### SYSTEM ROLE
        You are an analytical reasoning engine with access to a memory history. Your job is to process evidence fragments and generate a precise answer strictly following a requested format.

        ### RETRIEVED EVIDENCE (Context)
        {evidence}

        ### REFERENCE INFORMATION
        - Current Question Date: {instance.question_date} (Use this as "Today" for temporal calculations).

        ### REASONING INSTRUCTIONS (MUST READ)
        Before generating the final answer, process the evidence following these logic rules:

        1. **Conflict Resolution (Knowledge Update):**
        - If the evidence shows a change of state (e.g., "I moved to Paris" after "I live in London"), the most recent `timestamp` takes precedence.
        - If the items are cumulative (e.g., "I have a bike" and later "I bought another one"), SUM THEM UP unless the evidence explicitly states the previous one was sold or lost.

        2. **Temporal Reasoning:**
        - If the question asks for elapsed time, calculate: `Current Question Date` - `Event Date`.
        - Explicitly look for past events that match the description (e.g., "museum with a friend") and ignore events that do not match exactly.

        3. **Preference Inference:**
        - If the user asks for recommendations, scan the evidence for brands, models, or styles they already own or like (e.g., if they have "Sony" gear, recommend "Sony" compatible items).

        4. **Strict Abstention:**
        - If looking for a specific data point (e.g., "30-gallon tank") and the evidence only mentions other values (18 or 20 gallons), DO NOT HALLUCINATE. Your answer must declare the lack of information.

        ### FINAL FORMAT INSTRUCTION (CRITICAL)
        Your final response must STRICTLY obey this instruction:
        "{format_instruction}"

        ### USER QUESTION
        {instance.question}

        ### YOUR ANSWER:
        """
        messages = [{"role": "user", "content": prompt}]
        answer = self.model.reply(messages)
        return answer
    
    def answer_format(self, question):
        prompt = f"""
        ### TASK
        Analyze the User Question and determine the precise formatting instructions for the final answer.
        DO NOT answer the question. Output ONLY the "Format Instruction".

        ### EXAMPLES

        Input: "How long is my commute?"
        Instruction: Answer only with the time quantity and unit (e.g., "45 minutes"). Be extremely concise.

        Input: "How many bikes do I have?"
        Instruction: Answer only with the total number (e.g., "2" or "2 bikes"). Perform a summation if there are multiple mentions.

        Input: "Date: 6/25. How many months have passed since the last time I...?"
        Instruction: Calculate the time difference relative to the current date. Answer only with the number and unit (e.g., "5 months").

        Input: "Can you recommend accessories for my setup?"
        Instruction: Answer with a detailed and justified list, strictly prioritizing the user's implicit preferences (brands, previous models owned) found in the context.

        Input: "How many 30-gallon tanks do I have?"
        Instruction: If there is no exact evidence for that specific size, state: "I do not have information regarding 30-gallon tanks."

        ### CURRENT INPUT
        Question: "{question}"

        ### INSTRUCTION:
        """
        
        response = [{"role": "user", "content": prompt}]
        answer = self.model.reply(response)
        return answer
    