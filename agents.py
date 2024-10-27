import os
from groq import Groq

class Coder():
    def __init__(self, client, problem):
        self.client = client
        self.problem = problem
        self.history = [{"role": "user",
                         "content": f"Você é um desenvolvedor Python e cientista de dados. Sua função é escrever código para resolver o seguinte problema de ciência de dados: {self.problem}. Seja conciso e certifique-se de documentar o seu código."}]

    def process_data(self):
        message = "Faça o seguinte no código: limpe, transforme e prepare os dados para análise."

        self.history.append({"role": "user",
                             "content": message})
        
        answer = self.client.chat.completions.create(
            messages = self.history,
            model = "llama3-8b-8192"
        )

        self.history.append({"role": "assistant",
                             "content": answer})
        
        return answer
    

    def analyze_data(self):
        message = "Faça o seguinte no código: realize análises estatísticas ou construa modelos de aprendizado de máquina."

        self.history.append({"role": "user",
                             "content": message})
        
        answer = self.client.chat.completions.create(
            messages = self.history,
            model = "llama3-8b-8192"
        )

        self.history.append({"role": "assistant",
                             "content": answer})
        
        return answer
    

    def visualize_results(self):
        message = "Faça o seguinte no código: gere gráficos e visualizações dos dados."

        self.history.append({"role": "user",
                             "content": message})
        
        answer = self.client.chat.completions.create(
            messages = self.history,
            model = "llama3-8b-8192"
        )

        self.history.append({"role": "assistant",
                             "content": answer})
        
        return answer
    

    def interpret_analysis(self):
        message = "Faça o seguinte no código: interprete os resultados, produzindo um texto que inclua os resultados da análise na forma de figuras e tabelas."

        self.history.append({"role": "user",
                             "content": message})
        
        answer = self.client.chat.completions.create(
            messages = self.history,
            model = "llama3-8b-8192"
        )

        self.history.append({"role": "assistant",
                             "content": answer})
        
        return answer