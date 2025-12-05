import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.datasets.LongMemEvalDataset import LongMemEvalInstance
from litellm import embedding


def embed_text(message, embedding_model_name):
    response = embedding(model=embedding_model_name, input=message)
    return response.data[0]["embedding"]


def get_messages_and_embeddings_and_dates(instance: LongMemEvalInstance, embedding_model_name):
    cache_path = f"data/rag/embeddings_{embedding_model_name.replace('/', '_')}/{instance.question_id}.parquet"
    #cache
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)

        messages = df["messages"].tolist()
        embeddings = df["embeddings"].tolist()
        dates = df["dates"].tolist()
        session_ids = df["sessions"].tolist()

        session_lookup = {s.session_id: s for s in instance.sessions}
        sessions = [session_lookup[sid] for sid in session_ids]

        return messages, embeddings, dates, sessions

    messages = []
    embeddings = []
    sessions = []
    dates = []

    for session in tqdm(instance.sessions, desc="Embedding sessions"):
        for message in session.messages:
            messages.append(message)
            sessions.append(session)
            embeddings.append(embed_text(
                f"date: {session.date}: {message['role']}: {message['content']}",
                embedding_model_name
            ))
            dates.append(session.date)

    # crear carpeta si no existe
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # guardar tipos serializables (NO objetos Python)
    df = pd.DataFrame({
        "messages": messages,
        "sessions": [s.session_id for s in sessions],
        "embeddings": embeddings,
        "dates": dates
    })
    df.to_parquet(cache_path)

    return messages, embeddings, dates, sessions



def retrieve_most_relevant_messages(instance: LongMemEvalInstance, k: int, embedding_model_name):

    question_embedding = embed_text(f"date: {instance.question_date}: {instance.question}", embedding_model_name)
    messages, embeddings, dates, sessions = get_messages_and_embeddings_and_dates(instance, embedding_model_name)

    similarity_scores = np.dot(embeddings, question_embedding)
    #print(similarity_scores)
    most_relevant_messages_indices = np.argsort(similarity_scores)[::-1][:k]
    most_relevant_messages = []
    most_relevant_dates = []
    most_relevant_sessions = []

    for i in most_relevant_messages_indices:
        most_relevant_messages.append(messages[i])
        most_relevant_dates.append(dates[i])
        most_relevant_sessions.append(sessions[i])
    for i in range(k):
        most_relevant_messages[i] = f"Date: {most_relevant_dates[i]}. Role: {most_relevant_messages[i]["role"]}. Message: {most_relevant_messages[i]["content"]} \n"
    return most_relevant_messages, most_relevant_sessions, most_relevant_dates


class RAGAgent:
    def __init__(self, model, embedding_model_name):
        self.model = model
        self.embedding_model_name = embedding_model_name

    def answer(self, instance: LongMemEvalInstance):
        topk_relevant_messages, topk_relevant_sessions, topk_relevant_dates = retrieve_most_relevant_messages(instance, 10, self.embedding_model_name)


        contextualized_topk_relevant_messages = self.retrieve_contextualized_messages(instance, topk_relevant_messages, topk_relevant_sessions)
        prompt = f"""
        You are a helpful assistant that answers a question based on the evidence.
        The evidence is: {contextualized_topk_relevant_messages}
        The question date is: {instance.question_date} and the question is: {instance.question}
        Return the answer to the question.
        """
        messages = [{"role": "user", "content": prompt}]
        answer = self.model.reply(messages)
        
        return answer
    
    def retrieve_contextualized_messages(self, instance: LongMemEvalInstance, topk_relevant_messages, topk_relevant_sessions):
        topk_contextualized_messages = []
        for i in range(len(topk_relevant_messages)):
            message = topk_relevant_messages[i]
            session = topk_relevant_sessions[i]
            context = self.contextualize_message_by_session(message, session)
            contextualized_message = f"{context}. {message}"
            topk_contextualized_messages.append(contextualized_message)

        return topk_contextualized_messages

    def contextualize_message_by_session(self, message, session):
        session_messages = session.messages
        session_messages_formatted = []
        for i in range(len(session_messages)):
            session_messages_formatted = f"Role: {session_messages[i]["role"]}. Message: s{session_messages[i]["content"]} \n"
        session_summary = f"Session date: {session.date}, session_messages: {session_messages_formatted}"
        prompt = f"""
        You will generate a brief context (1–2 sentences) describing where a message fits within a session.
        Do NOT repeat the message itself. Do NOT output metadata tokens like <unused045>.

        <chunk>
        {message}
        </chunk>

        <session_summary>
        {session_summary}
        </session_summary>
        Return ONLY the context, nothing else.
        """

        messages = [{"role": "user", "content": prompt}]
        context = self.model.reply(messages)
        return context.strip()
