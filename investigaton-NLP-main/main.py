import argparse
import json
import time
import os
from dotenv import load_dotenv
from src.models.LiteLLMModel import LiteLLMModel
from src.agents.JudgeAgent import JudgeAgent
from src.agents.RAGAgent import RAGAgent
from src.agents.SemanticRetrieverAgent import SemanticRetrieverAgent
from src.agents.ContextualizerAgent import ContextualizerAgent
from src.datasets.LongMemEvalDataset import LongMemEvalDataset
from src.models.QwenModel import QwenModel
from config.config import Config


load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Run LongMemEval evaluation pipeline")
    parser.add_argument(
        "--memory-model",
        type=str,
        default="ollama/gemma3:4b",
        help="Model name for memory/RAG agent (default: ollama/gemma3:4b)"
    )
    parser.add_argument(
        "--contextualizer-model",
        type=str,
        default="ollama/gemma3:4b",
        help="Model name for contextualizer agent"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="openai/gpt-5-mini",
        help="Model name for judge agent (default: openai/gpt-5-mini)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="short",
        choices=["oracle", "short"],
        help="Dataset type: oracle, short (default: short)"
    )
    parser.add_argument(
        "--dataset-set",
        type=str,
        default="longmemeval",
        choices=["longmemeval", "investigathon_evaluation", "investigathon_held_out"],
        help="Dataset set to use (default: longmemeval)"
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=500,
        help="Number of samples to process (default: 10)"
    )
    return parser.parse_args()


args = parse_args()

config = Config(
    memory_model_name=args.memory_model,
    judge_model_name=args.judge_model,
    longmemeval_dataset_type=args.dataset_type,
    longmemeval_dataset_set=args.dataset_set,
    contextualizer_model_name=args.contextualizer_model,
    N=args.num_samples,
)

print(f"\nInitializing models...")
print(f"  Memory Model: {config.memory_model_name}")
print(f"  Judge Model: {config.judge_model_name}")
print(f"  Embedding Model: {config.embedding_model_name}")
#Modelos
memory_model = LiteLLMModel(config.memory_model_name)
judge_model = LiteLLMModel(config.judge_model_name)
contextualizer_model = LiteLLMModel(config.contextualizer_model_name)

#Agentes
judge_agent = JudgeAgent(model=judge_model)
semantic_retriever_agent = SemanticRetrieverAgent(embedding_model_name=config.embedding_model_name)
contextualizer_agent = ContextualizerAgent(model=contextualizer_model)
memory_agent = RAGAgent(model=memory_model, semantic_retriever_agent=semantic_retriever_agent, contextualizer_agent = contextualizer_agent)

longmemeval_dataset = LongMemEvalDataset(config.longmemeval_dataset_type, config.longmemeval_dataset_set)

# Create results directory
results_dir = f"data/results/pruebaisenoughSINabstentions"
os.makedirs(results_dir, exist_ok=True)

print(f"\nResults will be saved to: {results_dir}")
print(f"Processing samples...")
print("=" * 100)

# Process samples
for instance in longmemeval_dataset[: config.N]:
    result_file = f"{results_dir}/{instance.question_id}.json"
    if os.path.exists(result_file):
        print(f"Skipping {instance.question_id} because it already exists", flush=True)
        continue
    ## se calcula la respuesta
    start_time = time.perf_counter()
    predicted_answer = memory_agent.answer(instance)
    sessions_used_by_question = memory_agent.get_sessions_used_by(instance.question_id)
    
    end_time = time.perf_counter()

    print("Turno de GPT5-Mini")
    #se verifica si la respuesta fue correcta
    if config.longmemeval_dataset_set != "investigathon_held_out":
        answer_is_correct = judge_agent.judge(instance, predicted_answer)

    # Save result
    with open(result_file, "w", encoding="utf-8") as f:
        result = {
            "question_id": instance.question_id,
            "question": instance.question,
            "predicted_answer": predicted_answer,
            "question_type": instance.question_type,
            "question_date": instance.question_date,
            "time_taken_to_answer": end_time-start_time,
            "session_ids_used": sessions_used_by_question
            
        }
        if config.longmemeval_dataset_set != "investigathon_held_out":
            result["answer"] = instance.answer
            result["answer_is_correct"] = answer_is_correct

        json.dump(result, f, indent=2)

        print(f"  Question: {instance.question}...")
        print(f"  Predicted: {predicted_answer}")
        if config.longmemeval_dataset_set != "investigathon_held_out":
            print(f"  Ground Truth: {instance.answer}")
            print(f"  Correct: {answer_is_correct}")
        print("-" * 100)

print("EVALUATION COMPLETE")
