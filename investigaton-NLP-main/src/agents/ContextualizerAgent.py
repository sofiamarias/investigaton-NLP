class ContextualizerAgent:
    def __init__(self, model):
        self.model = model
    
    def retrieve_contextualized_messages(self, messages, sessions):
        topk_contextualized_messages = []
        batch_messages = []
        for i, (msg, session) in enumerate(zip(messages, sessions)):
            for j in range(len(session.messages)):
                if(session.messages[j]['content'] == msg['content']):
                    message_index = j
                    break
            previous_messages = session.messages[message_index-3:message_index]
            context = " | ".join([f"{m["role"]}: {m["content"]}" for m in previous_messages])
            batch_messages.append({
                "id": i,
                "context": context,
                "target_message": msg["content"],
                "date": session.date
            })

        prompt = self.build_qwen_batch_prompt(batch_messages)
        print(prompt)
        response = self.model.reply([{"role": "user", "content": prompt}])

        #parsed_response = self.parse_batch_context(response)
        print(response)
        return topk_contextualized_messages

 

    def build_qwen_batch_prompt(self, batch_messages):
        prompt = """You are a Context Extraction Engine.
        Your task is to identify implicit information in the "Target Message" based on the "Conversation History".
        Return ONLY the missing entities, locations, or topics needed to disambiguate the message.

        ### RULES:
        1. **Extract ONLY the Missing Part**: Do NOT rewrite the message. Just output the missing noun phrase or entity.
        2. **Self-Contained Messages**: If the target message requires no extra context, return an empty string "".
        3. **Strict JSON**: Output ONLY a valid JSON list.

        ### EXAMPLES:
        - History: ["User: How is the weather in London?"] | Target: "And in Paris?" -> Output: "weather"
        - History: ["User: I like apples."] | Target: "Me too." -> Output: "likes apples"
        - History: ["User: Hello."] | Target: "Hi there." -> Output: ""

        ### OUTPUT FORMAT:
        [
        {"id": 0, "context": "weather in Paris"},
        {"id": 1, "context": ""}
        ]

        ### INPUT DATA:
        """
        for message in batch_messages:
            prompt += f"""
        ---
        ID: {message['id']}
        History: [{message['context']}]
        Target: "{message['target_message']}"
        """
            
        prompt += "\n\nGenerate the JSON output now:"
        return prompt
