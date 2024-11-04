import os
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": """Execute este código: # Este script calcula a média de uma lista de números fornecida pelo usuário

# Função para calcular a média
def calcular_media(numeros):
    # Verifica se a lista não está vazia
    if len(numeros) == 0:
        return 0
    # Soma todos os números e divide pela quantidade para obter a média
    return sum(numeros) / len(numeros)

# Solicita ao usuário para inserir os números, separados por espaços
entrada = input("Digite uma lista de números separados por espaço: ")

# Converte a entrada em uma lista de números (float)
numeros = [float(num) for num in entrada.split()]

# Calcula a média usando a função
media = calcular_media(numeros)

# Exibe o resultado
print(f"A média dos números fornecidos é: {media}")""",
        }
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)