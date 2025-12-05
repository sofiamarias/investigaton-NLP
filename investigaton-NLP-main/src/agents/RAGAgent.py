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
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        return df["messages"].tolist(), df["embeddings"].tolist(), df["dates"].tolist()

    messages = []
    embeddings = []
    dates = []
    for session in tqdm(instance.sessions, desc="Embedding sessions"):
        for message in session.messages:
            messages.append(message)
            embeddings.append(embed_text(f"date: {session.date}: {message['role']}: {message['content']}", embedding_model_name))
            dates.append(session.date)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    pd.DataFrame({"messages": messages, "embeddings": embeddings, "dates": dates}).to_parquet(cache_path)
    return messages, embeddings, dates


def retrieve_most_relevant_messages(instance: LongMemEvalInstance, k: int, embedding_model_name):

    question_embedding = embed_text(f"date: {instance.question_date}: {instance.question}", embedding_model_name)
    messages, embeddings, dates = get_messages_and_embeddings_and_dates(instance, embedding_model_name)

    similarity_scores = np.dot(embeddings, question_embedding)
    #print(similarity_scores)
    most_relevant_messages_indices = np.argsort(similarity_scores)[::-1][:k]
    most_relevant_messages = [messages[i] for i in most_relevant_messages_indices]
    most_relevant_dates = [dates[i] for i in most_relevant_messages_indices]
    for i in range(k):
        most_relevant_messages[i] = f"Date: {most_relevant_dates[i]}. Role: {most_relevant_messages[i]["role"]}. Message: {most_relevant_messages[i]["content"]} \n"
    return most_relevant_messages


class RAGAgent:
    def __init__(self, model, embedding_model_name):
        self.model = model
        self.embedding_model_name = embedding_model_name

    def answer(self, instance: LongMemEvalInstance):
        most_relevant_messages = retrieve_most_relevant_messages(instance, 10, self.embedding_model_name)
        
        promptToSummarize = f"""
        ### SYSTEM ROLE
You are an expert Information Curator and Context Optimizer. Your goal is to prepare a sanitized, high-density information packet for a downstream reasoning model (Gemma3:4b).

### INPUT DATA
1. **SOURCE_MESSAGES**: {most_relevant_messages}
2. **USER_QUERY**: {instance.question}
3. **CURRENT_DATE**: {instance.question_date}

### INSTRUCTIONS
Analyze the `SOURCE_MESSAGES` against the `USER_QUERY` and execute the following filtering process:

1.  **Temporal Validation**: Check the timestamps of the messages against the `CURRENT_DATE`. Discard information that is explicitly outdated or superseded by more recent messages in the list.
2.  **Noise Elimination**: Remove all conversational fillers (e.g., "Hello," "How are you," "Thanks"), metadata irrelevant to the content, and formatting noise.
3.  **Fact Extraction**: Extract *only* the specific sentences or data points strictly necessary to answer the `USER_QUERY`. If a message contains no relevant information, ignore it completely.
4.  **Multimodal Awareness**: If the text describes images or visual data relevant to the query, include those descriptions clearly.

### OUTPUT FORMAT
Return **only** an output following this format. It should be a list of the following format with
enough information to answer the question accurately. Do not include anything else.

  "relevant_context":
      "date": "Date of the source (optional)",
      "fact": "The specific extracted information."
        """
        messageToSummarize = [{"role": "user", "content": promptToSummarize}]
        summarizedAnswer = self.model.reply(messageToSummarize)
        prompt = f"""
        You are a helpful assistant that answers a question based on the evidence.
        The evidence is: {summarizedAnswer}
        The question date is: {instance.question_date} and the question is: {instance.question}
        Return the answer to the question.
        """
        messages = [{"role": "user", "content": prompt}]
        answer = self.model.reply(messages)
        print(summarizedAnswer)
        return answer
