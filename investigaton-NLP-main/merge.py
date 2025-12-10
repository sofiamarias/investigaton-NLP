import os
import json
import pandas as pd
from src.datasets.LongMemEvalDataset import LongMemEvalDataset

set = "investigathon_evaluation"
type = "oracle"

results_dir = (
    f"data/results/prueba8ParesConPlanINVESTIGATHONNOHELDOUT"
)
json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]

results = []
for filename in json_files:
    filepath = os.path.join(results_dir, filename)
    with open(filepath, "r") as f:
        data = json.load(f)
        results.append(data)
df = pd.DataFrame(results)

longmemeval_df = LongMemEvalDataset(type, set).dataset

df.merge(longmemeval_df[["question_id", "question_type"]], on="question_id", how="inner").groupby("question_type").agg(
    answer_is_correct_mean=("answer_is_correct", "mean"), count=("answer_is_correct", "count")
).sort_values(by="answer_is_correct_mean", ascending=False).reset_index()
