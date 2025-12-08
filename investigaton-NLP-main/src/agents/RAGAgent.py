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
        for document in topk_relevant_documents:
            contents.append(f"[{document['timestamp']}]: {document['content']}")
            self.sessions_id_used_by_question.setdefault(instance.question_id, []).append(document['session_id'])        #Todavia no funciona
        #list_contextualized_topk_relevant_messages = self.contextualizer_agent.retrieve_contextualized_messages(topk_relevant_messages, topk_relevant_sessions)
        #contextualized_topk_relevant_messages = "\n\n".join(list_contextualized_topk_relevant_messages)

        answer_format = self.answer_format(instance.question)
        prompt = f"""
        ### INSTRUCTION:
        You are an intelligent assistant. You must answer the QUESTION based on the provided MEMORY CONTEXT.

        ### MEMORY CONTEXT:
        {contents}
        -----------------------
        ### GUIDELINES:
        1. **Analyze**: Read the context carefully. Look for direct answers OR information that allows you to infer the answer.
        2. **Synthesize**: You can combine information from multiple messages to form a complete answer.
        3. **Strictness**: If the context is not related to the question, say "I don't have enough information."
        4. **Temporal Reasoning**: Pay close attention to the dates in brackets [YYYY-MM-DD]. If you find contradictory information, ALWAYS prioritize the information with the most recent date.

        ### USER QUESTION:
        [{instance.question_date}]: {instance.question}

        ###Output Format
        {answer_format}
        
        Answer ONLY the question using the given format.
        
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
    