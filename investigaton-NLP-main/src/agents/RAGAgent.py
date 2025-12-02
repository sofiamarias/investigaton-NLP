import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.datasets.LongMemEvalDataset import LongMemEvalInstance
from litellm import embedding


def embed_text(message, embedding_model_name):
    response = embedding(model=embedding_model_name, input=message)
    return response.data[0]["embedding"]


def get_messages_and_embeddings(instance: LongMemEvalInstance, embedding_model_name):
    cache_path = f"data/rag/embeddings_{embedding_model_name.replace('/', '_')}/{instance.question_id}.parquet"
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        return df["messages"].tolist(), df["embeddings"].tolist()

    messages = []
    embeddings = []
    for session in tqdm(instance.sessions, desc="Embedding sessions"):
        for message in session.messages:
            messages.append(message)
            embeddings.append(embed_text(f"{message['role']}: {message['content']}", embedding_model_name))

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    pd.DataFrame({"messages": messages, "embeddings": embeddings}).to_parquet(cache_path)
    return messages, embeddings


def retrieve_most_relevant_sessions(instance: LongMemEvalInstance, k: int, embedding_model_name):

    question_embedding = embed_text(instance.question, embedding_model_name)
    messages, embeddings = get_messages_and_embeddings(instance, embedding_model_name)

    similarity_scores = np.dot(embeddings, question_embedding)
    most_relevant_messages_indices = np.argsort(similarity_scores)[::-1][:k]
    most_relevant_messages = [messages[i] for i in most_relevant_messages_indices]

    return most_relevant_messages


class RAGAgent:
    def __init__(self, model, embedding_model_name):
        self.model = model
        self.embedding_model_name = embedding_model_name

    def answer(self, instance: LongMemEvalInstance):
        most_relevant_sessions = retrieve_most_relevant_sessions(instance, 10, self.embedding_model_name)

        prompt = f"""
        You are a helpful assistant that answers a question based on the evidence.
        The evidence is: {most_relevant_sessions}
        The question is: {instance.question}
        Return the answer to the question.
        """
        promptSinEvidencia = f"""
        Return the answer to this question {instance.question}.
        
        """
        messages = [{"role": "user", "content": prompt}]
        reply = self.model.reply(messages)
        answer = reply.choices[0].message.content
        
        return answer
