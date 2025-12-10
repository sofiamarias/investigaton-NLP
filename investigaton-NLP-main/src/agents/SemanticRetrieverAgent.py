import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from src.datasets.LongMemEvalDataset import LongMemEvalInstance
from litellm import embedding
from datetime import datetime
"""
Este agente se encarga de, dada una instancia del benchmark, con su query y sus sessions,
conseguir los top-k mensajes más relevantes.(Pensamos en filtrar primero por sesiones)
"""
class SemanticRetrieverAgent:
    def __init__(self, embedding_model_name, reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.embedding_model_name = embedding_model_name
        print(f"Cargando modelo Cross-Encoder: {reranker_model_name}...")
        self.reranker = CrossEncoder(reranker_model_name)

    #Calcular el embedding de un texto
    def embed_text(self, message, embedding_model_name):
        response = embedding(model=embedding_model_name, input=message)
        return response.data[0]["embedding"]
    
    def answer(self, instance: LongMemEvalInstance):
        pass
    ##

    def format_date(self, date):
        dt = datetime.strptime(date, "%Y/%m/%d (%a) %H:%M")
         # Armado del formato final
        month = dt.strftime("%B")         
        day = dt.strftime("%d")          
        year = dt.strftime("%Y")
        weekday = dt.strftime("%A")      
        time = dt.strftime("%H:%M")       

        return f"{month} {day} {year}, ({weekday}) {time}"
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
            session_date = self.format_date(session.date)
            for i in range(total_msgs-1):
                msg_current = msgs[i]
                msg_next = msgs[i+1]

                #Por si no se respeta el invariante user->assistant, filtramos
                if msg_current['role'] == 'user' and msg_next['role'] == 'assistant':

                    round_id = f"{session.session_id}_round_{i}"
                    #Cambiamos a chunk por mensaje.
                    user_text = msg_current['content']
                    assistant_text = msg_next['content']
                    
                    #Guardamos para el re-ranking
                    combined_rerankingtext = (
                        f"User: {user_text}\n"
                        f"Assistant: {assistant_text}"
                    )

                    doc_user = {
                        "id": f"{round_id}_user",
                        "embedding": self.embed_text(f"[{session_date}]: {user_text}", self.embedding_model_name),
                        "content": user_text,
                        #Metadatos
                        "role": "user",
                        "session_id": session.session_id,
                        "timestamp": session.date,
                        "round_id": round_id,
                        "partner_content": assistant_text,
                        "full_pair_text": combined_rerankingtext
                    }

                    doc_assistant = {
                        "id": f"{round_id}_assistant",
                        "embedding": self.embed_text(f"[{session_date}]: {assistant_text}", self.embedding_model_name),
                        "content": assistant_text,
                        #Metadatos
                        "role": "assistant",
                        "session_id": session.session_id,
                        "timestamp": session.date,
                        "round_id": round_id,
                        "partner_content": user_text,
                        "full_pair_text": combined_rerankingtext
                    }
                    documents.append(doc_user)
                    documents.append(doc_assistant)
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
        sorted_indices = np.argsort(similarity_scores)[::-1]
        candidate_docs = []
        seen_round_ids = set()
        candidate_scores = []
        #Re-ranking
        for idx in sorted_indices:
            if len(candidate_docs) >= bi_encoder_k:
                break

            doc = documents[idx]
            round_id = doc["round_id"]
            if round_id in seen_round_ids:
                continue
            seen_round_ids.add(round_id)
            candidate_docs.append(doc)
            candidate_scores.append(similarity_scores[idx])
        if not candidate_docs:
            return []
        
        #Obtenemos el ranking de cada documento
        pairs = []
        for doc in candidate_docs:
            context_text = doc["full_pair_text"]
            pairs.append([instance.question, context_text])
        
        reranked_scores = self.reranker.predict(pairs)
        #Rearmamos los documentos
        scored_candidates = list(zip(reranked_scores, candidate_docs))
        #Ordenamos
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        top_cross_encoder_k_tuples = scored_candidates[:cross_encoder_k]
        final_documents = [doc for score, doc in top_cross_encoder_k_tuples]
        #Ordenamos por fecha
        final_documents.sort(key=lambda x: x["timestamp"]) 
        return final_documents

