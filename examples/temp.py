from pydantic import BaseModel
from openai import OpenAI
from anthropic import Anthropic
import toolflow

#client = toolflow.from_openai(OpenAI())
client = toolflow.from_anthropic(Anthropic())
orginal_client = Anthropic()

test = orginal_client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "What is 12 divided by 2?"}
    ]
)

@toolflow.tool
def divide(a: int, b: int) -> int:
    print(f"Dividing {a} by {b}")
    return a / b

# completion = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant that can divide two numbers."},
#         {"role": "user", "content": "What is 12 divided by 0?"},
#     ],
#     tools=[divide]
# )

completion = client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is 12 divided by 2?"
                }
            ]
        }
    ],
    system="You are a helpful assistant that can divide two numbers.",
    tools=[divide]
)

print(completion)
