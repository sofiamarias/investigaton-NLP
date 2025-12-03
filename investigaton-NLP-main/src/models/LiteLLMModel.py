from litellm import completion
from src.models.Model import Model


class LiteLLMModel(Model):
    def __init__(self, name):
        super().__init__(name)

    def reply(self, messages, tools=None):
        response = completion(model=self.name, messages=messages)
        return response.choices[0].message.content
