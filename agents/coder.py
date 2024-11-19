import sys
import os
from agents.utils import *
from agents.parser import *

if __name__ == '__main__':
    # Obtém o diretório do arquivo atual
    current_dir = os.path.dirname(__file__)
    
    # Navega até o diretório pai do pai (ou seja, dois níveis acima)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    # Define o diretório pai do pai como diretório atual
    os.chdir(parent_dir)
    
    # Opcional: Adiciona o diretório pai do pai ao sys.path para permitir importações
    sys.path.append(parent_dir)

from policy import EpsilonGreedyPolicy

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


class Coder():
    def __init__(self, client, problem):
        self.client = client
        self.problem = problem
        self.actions = [self.process_data, self.analyze_data, self.visualize_results, self.interpret_analysis]
        self.policy = EpsilonGreedyPolicy(4)
        self.history = [{"role": "system",
                         "content": prompt_coder.format(problem = self.problem)}]
        self.code = None
        self.state = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.current_action = None

    def act(self):
        action = self.policy.get_action(self.state)
        self.current_action = action
        self.actions[action]()
    

    def update_policy(self, state, action, reward, next_state):
        self.policy.update(state, action, reward, next_state)


    def reset(self):
        self.history = [{"role": "system",
                         "content": prompt_coder.format(problem = self.problem)}]
        self.code = None
        self.state = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def _get_llm_response(self, prompt: str) -> str:
        try:
            self.history.append({"role": "user", "content": prompt})

            # Call the Groq client

            answer = self.client.chat.completions.create(
                messages = self.history,
                model = "llama3-8b-8192"
            ).choices[0].message.content
            print(answer)
            self.history.append({"role": "assistant", "content": answer})

            self.code = extract_code(answer)

        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return "Unable to get LLM feedback at this time."

    def process_data(self):
        self._get_llm_response(prompt_process_data)

    def analyze_data(self):
        self._get_llm_response(prompt_analyze_data)

    def visualize_results(self):
        self._get_llm_response(prompt_visualize_results)
    
    def interpret_analysis(self):
        self._get_llm_response(prompt_interpret_analysis)