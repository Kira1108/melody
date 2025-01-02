import ollama 
from dataclasses import dataclass
import re
import json

TURN_DETECT_PROMPT = """
# Task: Telecommunication Language Understanding.

# Context: A customer service assistant in automibile industry talking with a customer via telephone.


# Instruction
The customer's voice is pasrsed with a streamed speech-to-text model, which means that the current received text is not complete yet.
Your task is a determine whether the customer is finished his turn, waiting for a reply or still speaking based on the text only.

# Input
You may be provided a short piece of text or a chatting history(in this case, you need to determine whether the last message is finished or not).

# Output
You need output a valid json codeblock enclosed in triple backtics like ```json...```
with keys been "status", possible values are "speech" or "finished"

Example:
Input: 我想问一下，
Output: 
```json
{{
  "status": "speech"
}}
```

# ASR customer text:
{text}

# Result:"""


def extract_json_codeblock(json_string:str):
    pattern = r'```(?:json)?\n(.*?)\n```'
    matches = re.findall(pattern, json_string, re.DOTALL)
    matches = [json.loads(m) for m in matches]
    return matches

@dataclass
class TurnDetector:
    
    model_name:str = "qwen2.5:0.5b"
    def shutup(self, text:str) -> bool:
        return True
        prompt = TURN_DETECT_PROMPT.format(text = text)
        response = ollama.chat(model=self.model_name, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ])
        response = extract_json_codeblock(response['message']['content'])[0]
        return response['status'] == "finished"
    
if __name__ == "__main__":
    td = TurnDetector()
    print(td.shutup("你好，请问你们是义新公司么？"))