from src.datasets.LongMemEvalDataset import LongMemEvalInstance

class JudgeAgent:
    def __init__(self, model):
        self.model = model

    def judge(self, instance: LongMemEvalInstance, predicted_answer):
        prompt = f"""
        You are a helpful assistant that judges the correctness of an answer to a question.
        The question is: {instance.question}
        The memory agent answer is: {predicted_answer}
        The ground truth answer is: {instance.answer}
        Return True if the prediction is correct, False otherwise. No other text or explanation.
        """
        messages = [{"role": "user", "content": prompt}]
        judgment = self.model.reply(messages)
        return eval(judgment)
