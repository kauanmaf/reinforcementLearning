import os
import sys
from parser import *

if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    os.chdir(parent_dir)
    sys.path.append(parent_dir)

# Pegando o prompt de entrada
with open("prompts/judger.txt", "r") as file:
    prompt_judger = file.read()

# Classe do julgador, que avalia o relatório
class Judger():
    def __init__(self, client, problem):
        # Cliente do chat model
        self.client = client
        # Descrição do problema
        self.problem = problem
        # Histórico da conversa
        self.history = [{"role": "system",
                         "content": prompt_judger.format(problem = self.problem)}]
        
    # Função para avaliar o relatório
    def judge(self, report):
        # Inicializando a conversa com a descrição do problema e o relatório
        self.history = [{"role": "system",
                         "content": prompt_judger.format(problem = self.problem)},
                        {"role": "user",
                         "content": report}]

        # Inicializando a variável que armazenará a resposta
        answer = None

        try:
            # Enquanto não tiver uma resposta adequada do modelo...
            while not answer:
                # Pega a resposta do modelo
                answer = self.client.chat.completions.create(
                    messages = self.history,
                    model = "gemma-7b-it",
                    temperature = 0,
                    max_tokens = 65
                )
                answer = answer.choices[0].message.content

                # Tenta tirar a tupla de notas da resposta
                answer = parse_tuple(answer)

        except:
            return None
        
        return answer