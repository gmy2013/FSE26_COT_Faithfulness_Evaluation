# import List
import os

import openai
from openai import OpenAI
def generate_code(prompt: str, n: int = 1) -> list[str]:

    try:


        client = OpenAI(api_key="sk-bb1bfbb4610a43a7bbc1f8af799ec8ee", base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",  #  "gpt-4o-2024-05-13"
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that generates Python code. Complete the following function based on the given docstring."},
                {"role": "user", "content": prompt}
            ],
            n=n,
            temperature=0.8,
            stop=["\nclass", "\n#", "\nif __name__"]
        )
        completions = [choice.message.content.strip() for choice in response.choices]
        return completions
    except Exception as e:
        print(f"An error occurred during code generation: {e}")
        return [""] * n