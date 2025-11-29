import re
import json
from src.models.TransformersModel import TransformersModel


class QwenModel(TransformersModel):
    def extract_tool_calls(self, content):
        tool_calls = []
        pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"

        def call_replacer(match):
            call_data = json.loads(match.group(1))
            idx = len(tool_calls)
            tool_calls.append(
                {
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": call_data["name"],
                        "arguments": json.dumps(call_data["arguments"]),
                    },
                }
            )
            return ""  # Remove the tool_call block

        # Remove all tool_call sections and collect them into tool_calls
        cleaned_content = re.sub(pattern, call_replacer, content, flags=re.DOTALL).strip()

        return tool_calls, cleaned_content

    def extract_thinking(self, content):
        thinking_blocks = [
            block.strip()
            for block in re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
            if block.strip()
        ]

        cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        reasoning_content = "\n\n".join(thinking_blocks) if thinking_blocks else None

        return reasoning_content, cleaned_content
