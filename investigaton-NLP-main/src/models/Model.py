from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def reply(self, messages, tools=None):
        raise NotImplementedError("Not implemented")

    def extract_tool_calls(self, content):
        raise NotImplementedError("Not implemented")

    def extract_thinking(self, content):
        raise NotImplementedError("Not implemented")

    def parse_response(self, content):
        raise NotImplementedError("Not implemented")
