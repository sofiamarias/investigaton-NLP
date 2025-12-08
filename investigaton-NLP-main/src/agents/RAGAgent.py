import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.datasets.LongMemEvalDataset import LongMemEvalInstance

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
        #Guardamos las sesiones usadas

        
        contents = []
        for i, document in enumerate(topk_relevant_documents):
            contents.append(f"Evidence N°{i}: [{document['timestamp']}]: {document['content']} \n")
            self.sessions_id_used_by_question.setdefault(instance.question_id, []).append(f"{document['session_id']}: {document['id']}")        #Todavia no funciona
        #list_contextualized_topk_relevant_messages = self.contextualizer_agent.retrieve_contextualized_messages(topk_relevant_messages, topk_relevant_sessions)
        #contextualized_topk_relevant_messages = "\n\n".join(list_contextualized_topk_relevant_messages)

        answer_format = self.answer_format(instance.question)
        prompt = f"""
        ### INSTRUCTION
        You are an intelligent assistant. Answer the USER QUESTION based ONLY on the provided CONVERSATION HISTORY.

        ### CONVERSATION HISTORY
        (Format: [Date] Role: Message)
        {contents}
        -----------------------

        ### GUIDELINES
        1. **Evidence-Based**: The history contains past interactions between a user and an assistant. Use ONLY this information. Do not use external knowledge.
        2. **Temporal Logic**: Pay attention to dates [YYYY-MM-DD]. If facts conflict, ALWAYS prioritize the MOST RECENT information.
        3. **Strict Abstention**: If the answer is not in the history, output EXACTLY: "I do not have enough information".
        4. **Language**: Answer STRICTLY IN ENGLISH.

        ### USER QUESTION
        [{instance.question_date}]: {instance.question}
        Answer the question
        """
        messages = [{"role": "user", "content": prompt}]
        answer = self.model.reply(messages)
        return answer
    
    def answer_format(self, question):
        prompt = f"""
        ### INSTRUCTION:
        You are an expert Response Architect. Your goal is to provide a specific instruction on HOW to answer the user's question. 
        DO NOT answer the question. Only describe the ideal format.

        ### EXAMPLES:

        User Question: "How many days ago was my birthday?"
        Format Instruction: Answer with a single short sentence stating the time difference (e.g., "It was 5 days ago").

        User Question: "What was the last book I read?"
        Format Instruction: Answer with the title of the book only, without extra text or punctuation.

        User Question: "What do you recommend me for breakfast?"
        Format Instruction: Provide a polite recommendation based on the user's past preferences mentioned in the context.

        User Question: "List the python libraries mentioned."
        Format Instruction: Provide a bulleted list of the libraries.

        ### TASK:
        Define the Format Instruction for the following question.
        User Question: "{question}"

        Format Instruction:
        """
        response = [{"role": "user", "content": prompt}]
        answer = self.model.reply(response)
        return answer
    