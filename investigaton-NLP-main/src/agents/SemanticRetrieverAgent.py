import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.datasets.LongMemEvalDataset import LongMemEvalInstance
from litellm import embedding

"""
Este agente se encarga de, dada una instancia del benchmark, con su query y sus sessions,
conseguir los top-k mensajes más relevantes.(Pensamos en filtrar primero por sesiones)
"""
class SemanticRetrieverAgent:
    def __init__(self, embedding_model_name):
        self.embedding_model_name = embedding_model_name
    #Calcular el embedding de un texto
    def embed_text(self, message, embedding_model_name):
        response = embedding(model=embedding_model_name, input=message)
        return response.data[0]["embedding"]
    
    def answer(self, instance: LongMemEvalInstance):
        pass
    ##
    def get_messages_and_embeddings_and_dates(self, instance: LongMemEvalInstance):
        #cache
        cache_path = f"data/rag/embeddings_{self.embedding_model_name.replace('/', '_')}/{instance.question_id}.parquet"
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
                embeddings.append(self.embed_text(
                f"date: {session.date}: {message['role']}: {message['content']}",
                self.embedding_model_name
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

    def retrieve_most_relevant_messages(self, instance: LongMemEvalInstance, k: int, similarity_threshold=0.45):
        #Borro embedding de fecha
        question_embedding = self.embed_text(f"{instance.question}", self.embedding_model_name)
        messages, embeddings, dates, sessions = self.get_messages_and_embeddings_and_dates(instance)
        similarity_scores = np.dot(embeddings, question_embedding)
        
        #Abstention
        if np.max(similarity_scores) < similarity_threshold:
            return [], [], []

        # Recuperamos índices ordenados por relevancia
        top_indices = np.argsort(similarity_scores)[::-1][:k]

        found_items = []
        for i in top_indices:
            found_items.append({
                "message": messages[i],
                "date": dates[i],
                "session": sessions[i],
                "score": similarity_scores[i]
            })

        #Ordeno los mensajes por fecha
        found_items.sort(key=lambda x: x["date"]) 

        final_messages = []
        final_sessions = []
        final_dates = []
        
        for item in found_items:
            final_messages.append(item['message'])
            final_sessions.append(item['session'])
            final_dates.append(item['date'])

        return final_messages, final_sessions, final_dates

