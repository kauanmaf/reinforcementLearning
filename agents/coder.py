import sys
import os
from parser import *

if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    os.chdir(parent_dir)
    sys.path.append(parent_dir)

from policy import EpsilonGreedyPolicyApprox

# Pegando os prompts de entrada
with open("prompts/coder.txt", "r") as file:
    prompt_coder = file.read()
with open("prompts/analyze_data.txt", "r") as file:
    prompt_analyze_data = file.read()
with open("prompts/interpret_analysis.txt", "r") as file:
    prompt_interpret_analysis = file.read()
with open("prompts/process_data.txt", "r") as file:
    prompt_process_data = file.read()
with open("prompts/visualize_results.txt", "r") as file:
    prompt_visualize_results = file.read()

# Classe do agente codador
class Coder():
    def __init__(self, client, problem):
        # Cliente do chat model
        self.client = client
        # Descrição do problema
        self.problem = problem
        # Lista das ações do agente
        self.actions = [self.process_data, self.analyze_data, self.visualize_results, self.interpret_analysis]
        # Política do agente como instância da aproximação com epsilon greedy
        self.policy = EpsilonGreedyPolicyApprox(14, 4, model_path="models/coder_model.pth")
        # Histórico da conversa com o prompt inicial
        self.history = [{"role": "system",
                         "content": prompt_coder.format(problem = self.problem)}]
        # Código gerado
        self.code = None
        # Estado atual com base nas notas dadas pelo reviewer
        self.state = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        # Ação tomada nessa rodada
        self.current_action = None
    
    def __str__(self):
        if self.current_action is not None:
            action_names = ["processando dados", "analizando dados", "visualizando resultados", "análise interpretativa"]
            return f"Current action: {action_names[self.current_action]}"
        return "Current action: None"

    # Função de ação do agente codador
    def act(self):
        # Escolhendo a ação com base na política
        action = self.policy.get_action(self.state)
        # Salvando a ação escolhida
        self.current_action = action
        # Executando a ação
        self.actions[action]()
    
    # Função de atualização da política
    def update_policy(self, state, action, reward, next_state):
        # Atualiza a política com base nos estados atual e seguinte, na ação e na recompensa
        self.policy.update(state, action, reward, next_state)

    # Função para colocar o agente de volta no estado inicial
    def reset(self):
        # Resetando seu histórico
        self.history = [{"role": "system",
                         "content": prompt_coder.format(problem = self.problem)}]
        # Resetando seu código
        self.code = None
        # Voltando ao estado inicial
        self.state = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # Função para obter a resposta do LLM
    def _get_llm_response(self, prompt: str) -> str:
        try:
            if len(self.history) > 4:
                del self.history[1]
                del self.history[2]
                    
            # Adicionando o prompt ao histórico
            self.history.append({"role": "user", "content": prompt})

            print("oi c1")
            # Pegando a resposta do LLM
            answer = self.client.chat.completions.create(
                messages = self.history,
                model = "llama3-70b-8192"
            ).choices[0].message.content

            print("oi c2")
            
            # Salvando a resposta no histórico
            self.history.append({"role": "assistant", "content": answer})

            # Extraindo o código da resposta
            self.code = extract_code(answer)

        except Exception as e:
            print(f"Error getting LLM response: {e}")

    # Função para escrever código para processamento de dados
    def process_data(self):
        self._get_llm_response(prompt_process_data)

    # Função para escrever código para análise de dados
    def analyze_data(self):
        self._get_llm_response(prompt_analyze_data)

    # Função para escrever código para visualização de resultados
    def visualize_results(self):
        self._get_llm_response(prompt_visualize_results)
    
    # Função para escrever código para interpretação de análises
    def interpret_analysis(self):
        self._get_llm_response(prompt_interpret_analysis)