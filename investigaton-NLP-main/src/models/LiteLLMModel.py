from litellm import completion

from src.models.Model import Model


class LiteLLMModel(Model):
    def __init__(self, name):
        super().__init__(name)

    def reply(self, messages, temperature, tools=None):
        if self.name == 'ollama/gemma3:4b':
            response = completion(model=self.name, temperature=temperature, messages=messages, num_ctx = 15000)
        if self.name == 'openai/gpt-5-mini':
            response = completion(model=self.name, temperature=temperature, messages=messages)

        
        return response.choices[0].message.content
