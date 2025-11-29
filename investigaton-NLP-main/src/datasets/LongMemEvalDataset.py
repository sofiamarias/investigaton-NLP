import json
import pandas as pd
import os


class Session:
    def __init__(self, session_id, date, messages):
        self.session_id = session_id
        self.date = date
        self.messages = messages

    def __repr__(self):
        return f"Session(session_id={self.session_id}, date={self.date}, messages={self.messages})"


class LongMemEvalInstance:
    def __init__(self, question_id, question, sessions, t_question, answer):
        self.question_id = question_id
        self.question = question
        self.sessions = sessions
        self.t_question = t_question
        self.answer = answer

    def __repr__(self):
        return f"LongMemEvalInstance(question={self.question}, sessions={self.sessions}, t_question={self.t_question}, answer={self.answer})"


class LongMemEvalDataset:
    def __init__(self, type, set):

        if type not in ["oracle", "short"]:
            raise ValueError(f"Invalid dataset type: {type}. Must be 'oracle' or 'short'")

        if set not in ["longmemeval", "investigathon_evaluation", "investigathon_held_out"]:
            raise ValueError(
                f"Invalid dataset set: {set}. Must be 'longmemeval' or 'investigathon_evaluation' or 'investigathon_held_out'"
            )

        if set == "longmemeval":
            path = {
                "oracle": "data/longmemeval/longmemeval_oracle.json",
                "short": "data/longmemeval/longmemeval_s_cleaned.json",
            }[type]
        elif set == "investigathon_evaluation":
            path = {
                "oracle": "data/investigathon/Investigathon_LLMTrack_Evaluation_oracle.json",  # Solo sesiones relevantes
                "short": "data/investigathon/Investigathon_LLMTrack_Evaluation_s_cleaned.json",  # Todas las sesiones (~115k tokens)
            }[type]
        elif set == "investigathon_held_out":
            if type != "short":
                raise ValueError(f"Invalid dataset type: {type} for held-out set. Must be 'short'")
            path = "data/investigathon/Investigathon_LLMTrack_HeldOut_s_cleaned.json"
        else:
            raise ValueError(
                f"Invalid dataset set: {set}. Must be 'longmemeval' or 'investigathon_evaluation' or 'investigathon_held_out'"
            )

        with open(path, "r", encoding="utf-8") as f:
            self.dataset = pd.DataFrame(json.load(f)).sample(frac=1, random_state=42).reset_index(drop=True)

        self.current_index = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        sliced_data = self.dataset.iloc[key]
        if isinstance(key, slice):
            return [self.instance_from_row(row) for _, row in sliced_data.iterrows()]
        else:
            return self.instance_from_row(sliced_data.iloc[key])

    def instance_from_row(self, row):
        return LongMemEvalInstance(
            question_id=row["question_id"],
            question=row["question"],
            sessions=[
                Session(session_id=session_id, date=date, messages=messages)
                for session_id, date, messages in zip(
                    row["haystack_session_ids"], row["haystack_dates"], row["haystack_sessions"]
                )
            ],
            t_question=row["question_date"],
            answer=row.get('answer')
        )
