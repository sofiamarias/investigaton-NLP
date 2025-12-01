from src.datasets.LongMemEvalDataset import LongMemEvalInstance

class JudgeAgent:
    def __init__(self, model):
        self.model = model
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
    def judge(self, instance: LongMemEvalInstance, predicted_answer):
        prompt = f"""
        You are a helpful assistant that judges the correctness of an answer to a question.
        The question is: {instance.question}
        The memory agent answer is: {predicted_answer}
        The ground truth answer is: {instance.answer}
        Return True if the prediction is correct, False otherwise. No other text or explanation.
        """
        messages = [{"role": "user", "content": prompt}]
        reply = self.model.reply(messages)
        judgment = reply.choices[0].message.content
        self.total_tokens += reply.usage.total_tokens
        self.prompt_tokens += reply.usage.prompt_tokens
        self.completion_tokens += reply.usage.completion_tokens
        return eval(judgment)