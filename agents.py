import os
from groq import Groq
from policy import EpsilonGreedyPolicy
import numpy as np
import ast

class Coder():
    def __init__(self, client, problem):
        self.client = client
        self.problem = problem
        self.actions = [self.process_data, self.analyze_data, self.visualize_results, self.interpret_analysis]
        self.policy = EpsilonGreedyPolicy(4)
        self.history = [{"role": "system",
                         "content": f"Você é um desenvolvedor Python e cientista de dados. Sua função é escrever código para resolver o seguinte problema de ciência de dados: {self.problem}. Seja conciso e certifique-se de documentar o seu código."}]
        

    def act(self, state):
        action = self.policy.get_action(state)
        answer = self.actions[action]()
        
        return answer
    

    def update_policy(self, state, action, reward, next_state):
        self.policy.update(state, action, reward, next_state)


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
    

class Judger():
    def __init__(self, client, problem):
        self.client = client
        self.problem = problem
        self.history = [{"role": "system",
                         "content": f"""Você é um avaliador de relatórios crítico. Sua função é atribuir notas de forma estruturada para os relatórios passados, que tratam do seguinte problema: {self.problem}. Escreva APENAS (NÃO ESCREVA MAIS NADA ALÉM!!!) uma tupla com notas de 0 a 10 para cada um dos tópicos a seguir:

                         - Clareza: A descrição do problema é clara e compreensível?
                         - Acurácia: A descrição do problema é precisa e relevante?
                         - Completude: Todos os conjuntos de dados relevantes são descritos com detalhes suficientes?
                         - Análise de qualidade de dados: A qualidade e as características dos dados são discutidas (por exemplo, valores ausentes, outliers)?
                         - Visualização: Os dados são bem visualizados usando gráficos ou tabelas?
                         - Abordagem: A metodologia escolhida é adequada para resolver o problema de análise de dados?
                         - Justificativa: Há uma justificativa clara para o motivo pelo qual métodos ou técnicas específicas foram escolhidos?
                         - Implementação: A implementação da metodologia é descrita com precisão e detalhes?
                         - Precisão: Os resultados são precisos e consistentes com os objetivos da análise?
                         - Entendimento: Os resultados são interpretados e discutidos adequadamente?
                         - Visualização: Os resultados são visualizados de forma eficaz e fáceis de entender (por exemplo, gráficos, tabelas)?
                         - Resumo: A conclusão resume sucintamente as principais descobertas do relatório?
                         - Implicações: As implicações dos resultados são discutidas?
                         - Recomendações: Há recomendações acionáveis ou próximas etapas?
                         """}]
        
    def judge(self, report):
        self.history.append({"role": "user",
                             "content": report})
        
        answer = self.client.chat.completions.create(
            messages = self.history,
            model = "llama3-8b-8192",
            temperature = 0
        )

        self.history.append({"role": "assistant",
                             "content": answer})
        
        answer = ast.literal_eval(answer.choices[0].message.content)
        
        return answer
    

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

teste = Judger(client, "Teste")
answer = teste.judge("Teste")
print(type(answer))