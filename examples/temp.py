from pydantic import BaseModel
from openai import OpenAI
from openai.types.chat import ChatCompletion
import toolflow

client = toolflow.from_openai(OpenAI())
s = OpenAI()

response = s.chat.completions.create(
    model="o4-mini",
    reasoning_effort="medium",
    messages=[
        {"role": "user", "content": "What is 12 divided by 2?"}
    ],
    stream=True
)

print(response.choices[0].message.
