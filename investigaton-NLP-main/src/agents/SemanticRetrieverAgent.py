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


    def get_messages_and_embeddings(self, instance: LongMemEvalInstance):
        #Cache
        cache_path = f"data/rag/paired_embeddings_{self.embedding_model_name.replace('/', '_')}/{instance.question_id}.parquet"
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            return df.to_dict(orient="records")
        
        documents = []

        for session in tqdm(instance.sessions, desc="Processing sessions"):
            msgs = session.messages
            total_msgs = len(msgs)

            for i in range(total_msgs-1):
                msg_current = msgs[i]
                msg_next = msgs[i+1]

                #Por si no se respeta el invariante user->assistant, filtramos
                if msg_current['role'] == 'user' and msg_next['role'] == 'assistant':
                    #Armamos el texto combinado
                    combined_text = (
                        f"User: {msg_current['content']}\n"
                        f"Assistant: {msg_next['content']}"
                    )
                    #ID del par
                    pair_id = f"{session.session_id}_pair_{i}"
                    #Embedding del par
                    vector = self.embed_text(combined_text, self.embedding_model_name)

                    #Armamos el diccionario
                    doc = {
                        "id": pair_id,
                        "embedding": vector,
                        #Guardamos el texto combinado
                        "content": combined_text,
                        "session_id": session.session_id,
                        "timestamp": session.date,
                    }
                    documents.append(doc)
        #Guardado
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df = pd.DataFrame(documents)
        df.to_parquet(cache_path)
        return documents

    def retrieve_most_relevant_messages(self, instance: LongMemEvalInstance, bi_encoder_k = 50, cross_encoder_k = 10, similarity_threshold=0.45):
        #Borro embedding de fecha
        question_embedding = self.embed_text(f"{instance.question}", self.embedding_model_name)
        documents = self.get_messages_and_embeddings(instance)

        #Abstention
        if not documents:
            return []
        
        #Armamos la lista de embeddings
        embeddings_list = [d["embedding"] for d in documents]
        embeddings_matrix = np.array(embeddings_list)

        similarity_scores = np.dot(embeddings_matrix, question_embedding)
        #Abstention
        if np.max(similarity_scores) < similarity_threshold:
            return []

        #Recuperamos índices ordenados por relevancia
        top_indices = np.argsort(similarity_scores)[::-1][:k]

        #Re-ranking
        
        

        final_documents = [documents[i] for i in top_indices]
    
        final_documents.sort(key=lambda x: x["timestamp"]) 
        
        return final_documents

