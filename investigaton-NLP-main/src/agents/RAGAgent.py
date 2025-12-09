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

            contents.append(
                "{\n"
                f'  "timestamp": "{document["timestamp"]}",\n'
                f'  "content": "{document["content"].replace("\\n", " ")}"\n'
                "}"
            )
            self.sessions_id_used_by_question.setdefault(instance.question_id, []).append(f"{document['session_id']}: {document['id']}")        #Todavia no funciona
        #list_contextualized_topk_relevant_messages = self.contextualizer_agent.retrieve_contextualized_messages(topk_relevant_messages, topk_relevant_sessions)
        #contextualized_topk_relevant_messages = "\n\n".join(list_contextualized_topk_relevant_messages)

        answer_format = self.answer_format(instance.question)
        print(f"ANSWER FORMAT: {answer_format}")
        print(f"CONTENTS: {contents}")
        print(f"QUESTION: {instance.question}")
        prompt = f"""
        You are an expert AI assistant. Your task is to answer the user's QUESTION using ONLY the provided DOCUMENT. Do not use any external knowledge.
        
        ### INSTRUCTIONS:
        1. Answer the QUESTION based *only* on the facts within the DOCUMENT.
        2. Follow the format output
        3. If you have a contradiction between two dates, prioritize the MOST RECENT.
        3. If you cannot find the answer in the DOCUMENT, respond with: 'I don't have enough information'. However, if you have it, answer.
        
        DOCUMENT:
        ---
        {contents}
        ---
        QUESTION:
        ---
        Question Date: {instance.question_date}. Question: {instance.question}
        ---
        FORMAT:
        ---
        {answer_format}
        ---
        Answer:
        """
        messages = [{"role": "user", "content": prompt}]
        answer = self.model.reply(messages)
        return answer
    
    def answer_format(self, question):
        prompt = f"""
        ### TASK:
        Determine the best answering format for the User Question.
        Do not answer the question. Output ONLY the "Format Instruction".

        ### EXAMPLES:

        User Question: "What bike do I have?"
        Context: [2022: Bought red bike], [2024: Sold red bike, bought blue bike]
        Format Instruction: Answer that the user has a blue bike. Explicitly mention that this information supersedes the 2022 data based on the more recent 2024 timestamp.

        User Question: "How long ago did I visit Paris?"
        Context: [User visit: 2023-01-01], [Today: 2023-01-10]
        Format Instruction: Calculate the time difference relative to today (e.g., "9 days ago") instead of stating the raw date.

        User Question: "Recommend a breakfast."
        Context: [Preference: "I only eat vegan food"]
        Format Instruction: Provide a breakfast recommendation that strictly adheres to the "vegan" preference found in context.

        User Question: "List all my project names."
        Context: [Session A: "Project Alpha"], [Session B: "Project Beta"]
        Format Instruction: Create a bulleted list combining items from all sessions (Alpha and Beta).

        User Question: "What is my dog's name?"
        Context: [No mention of pets]
        Format Instruction: State clearly that there is no information about a dog in the provided history.
        
        User Question: "How much did I spend on tech?"
        Context: [Ordered a mouse for $50], [Bought a keyboard for $100 later]
        Format Instruction: Identify the individual costs mentioned (mouse and keyboard), sum them up manually, and answer with the final calculated total amount.

        ### INPUT:
        User Question: "{question}"
        Format Instruction:
"""
        
        response = [{"role": "user", "content": prompt}]
        answer = self.model.reply(response)
        return answer
    