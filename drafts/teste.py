import os
from groq import Groq
import numpy as np

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

import pandas as pd

# Dados fictícios
data = {
    "Product": ["Product A", "Product B", "Product C", "Product A", "Product B"],
    "Sales": [200, 150, 300, 400, 250],
    "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    "Quantity": [3, 2, 5, 7, 4]
}

# Criando o DataFrame
df = pd.DataFrame(data)


chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": f"Faça uma análise sobre esses dados: {df}",
        }
    ],
    model="gemma-7b-it",
)

print(chat_completion.choices[0].message.content)