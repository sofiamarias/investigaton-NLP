import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.models.Model import Model


class TransformersModel(Model):
    def __init__(
        self,
        name,
        quantized: bool = False,
    ) -> None:
        super().__init__(name)
        self.quantized = quantized
        self.tokenizer, self.model = self.load_base_model(quantized=quantized)

    def load_base_model(self, quantized: bool = False):
        if quantized:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.name, device_map="auto", quantization_config=bnb_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.name, device_map="auto", dtype=torch.float16
            )

        tokenizer = AutoTokenizer.from_pretrained(self.name)
        return tokenizer, model

    def extract_tool_calls(self, content):
        raise NotImplementedError("Not implemented")

    def extract_thinking(self, content):
        raise NotImplementedError("Not implemented")

    def parse_response(self, content):
        reasoning_content, content = self.extract_thinking(content)
        tool_calls, content = self.extract_tool_calls(content)
        return reasoning_content, tool_calls, content

    def reply(self, messages, tools=None):
        formatted_input = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        raw_response_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        reasoning_content, tool_calls, response_text = self.parse_response(raw_response_text)

        return response_text
