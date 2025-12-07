import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.datasets.LongMemEvalDataset import LongMemEvalInstance

class RAGAgent:
    def __init__(self, model, semantic_retriever_agent):
        self.model = model
        self.semantic_retriever_agent = semantic_retriever_agent

    def answer(self, instance: LongMemEvalInstance):
        topk_relevant_messages, topk_relevant_sessions, topk_relevant_dates = self.semantic_retriever_agent.retrieve_most_relevant_messages(instance, 10)
        if not topk_relevant_messages:
            return "I do not have enough information"

        contextualized_topk_relevant_messages = self.retrieve_contextualized_messages(topk_relevant_messages, topk_relevant_sessions, topk_relevant_dates)
        prompt = f"""
        You are a helpful assistant that answers a question based on the evidence.
        Nothing else.
        The evidence is: {contextualized_topk_relevant_messages}
        The question date is: {instance.question_date} and the question is: {instance.question}
        Return the answer to the question.
        """
        messages = [{"role": "user", "content": prompt}]
        answer = self.model.reply(messages)
        print("Turno de GPT5-Mini")
        return answer
    
    def retrieve_contextualized_messages(self, topk_relevant_messages, topk_relevant_sessions, topk_relevant_dates):
        topk_contextualized_messages = []
        for i in range(len(topk_relevant_messages)):
            message = topk_relevant_messages[i]
            session = topk_relevant_sessions[i]
            date = topk_relevant_dates[i]
            context = self.contextualize_message_by_session(message, session)
            contextualized_message = f"[{date}]: Context: {context}. '\n' Message: {message}"
            topk_contextualized_messages.append(contextualized_message)

        return topk_contextualized_messages

    def contextualize_message_by_session(self, message, session):
        session_messages = session.messages
        message_index = 0
        for i in range(len(session_messages)):
            if(session_messages[i]['content'] == message['content']):
                message_index = i
                break
        previous_messages = session_messages[message_index-5:message_index]
      
        session_summary = f"Session date: {session.date}, previous messages: {previous_messages}"
        prompt = f"""
        You will generate a brief context (1–2 sentences) contextualizing a message with its previous messages.
        Do NOT repeat the message itself. Do NOT output metadata tokens like <unused045>.
        <message>
        {message}
        </message>

        <session_summary>
        {session_summary}
        </session_summary>
        Return ONLY the context, nothing else.
        """

        messages = [{"role": "user", "content": prompt}]
        context = self.model.reply(messages)
        return context.strip()