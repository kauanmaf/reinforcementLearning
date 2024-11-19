import os
import sys

if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    os.chdir(parent_dir)
    sys.path.append(parent_dir)

    
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
import subprocess
from policy import EpsilonGreedyPolicyApprox
from dotenv import load_dotenv
import os
import re
import subprocess
import tempfile
from parser import *

from policy import EpsilonGreedyPolicyApprox

# Pegando os prompts que vamos usar
with open("prompts/review_code.txt", "r") as file:
    prompt_review_code = file.read()

with open("prompts/create_report.txt", "r") as file:
    prompt_create_report = file.read()

with open("prompts/init_reviewer.txt", "r") as file:
    prompt_init_reviewer = file.read()

load_dotenv()
# Access the environment variables
SCRIPTS_PATH = os.getenv("SCRIPTS_PATH")

# Criamos a classe de CodeReviewer a qual será usada para avaliar o código
class CodeReviewer:
    def __init__(self, client: str, problem, model: str = "gemma-7b-it"):
        # Iniciamos o cliente, o modelo e o problema
        self.groq_client = client
        self.model = model
        self.problem = problem
        # Ele não começa com nenhum código
        self.code = None

        # Setamos variáveis de histórico para o groq ter um histórico da conversa
        self.feedback_history_report = [{
                    "role": "system",
                    "content": prompt_init_reviewer + self.problem
                }]
        self.feedback_history_grades = [{
                    "role": "system",
                    "content": prompt_init_reviewer + self.problem
                }]
        
        # Começamos a política de epsilon greedy
        self.actions = [self.create_report, self.execute_and_score_code, self.review_code, self.static_analysis]
        self.policy = EpsilonGreedyPolicyApprox(14, 4)
        
        # Começamso nossas variávais para armezenar nossas grades
        self.report = None
        self.grades = {"grades_llm": (0,0,0,0,0,0,0,0,0,0), 
                       "ruff": 0, 
                       "mypy": 0, 
                       "bandit" : 0, 
                       "execution_score": 0}
        self.current_action = None
        self.state = (0,0,0,0,0,0,0,0,0,0,0,0,0,0)

    # Função para voltar o CodeReviewer para o estado zero, deixando apenas a política aprendida até então
    def reset(self):
        self.feedback_history_report = [{
                    "role": "system",
                    "content": prompt_init_reviewer + self.problem
                }]
        self.feedback_history_grades = [{
                    "role": "system",
                    "content": prompt_init_reviewer + self.problem
                }]
        self.code = None
        self.report = None
        self.grades = {"grades_llm": (0,0,0,0,0,0,0,0,0,0), 
                       "ruff": 0, 
                       "mypy": 0, 
                       "bandit" : 0, 
                       "execution_score": 0}
        self.current_action = None
        self.state = (0,0,0,0,0,0,0,0,0,0,0,0,0,0)

    # Função que transforma uma lista com notas em um tupla de notas
    def get_coder_state_from_grades(self):
        return (*self.grades["grades_llm"], 
                self.grades["ruff"], 
                self.grades["mypy"], 
                self.grades["bandit"], 
                self.grades["execution_score"])
    
    # Função padrão para o reviewer pedir uma resposta do LLM
    def _get_llm_response(self, prompt: str, temperature: float = 0.7, review_code_bool = False) -> str:
        try:
            # Foi criado um booleno para saber de qual histórico ele tem que pegar o contexto
            if review_code_bool:
                # Adicionamos a message no histórico
                self.feedback_history_grades.append({"role": "user", "content": prompt})

                # Chamamos o groq para um resposta
                completion = self.groq_client.chat.completions.create(
                    messages=self.feedback_history_grades,
                    model=self.model,
                    temperature=temperature,
                    max_tokens=1000,
                )

                # Coloco então a resposta no histórico
                self.feedback_history_grades.append({
                    "role": "assistant",
                    "content": completion.choices[0].message.content
                })
                # Retorno a resposta
                return completion.choices[0].message.content

            else:
                # Adicionamos a message no histórico
                self.feedback_history_report.append({"role": "user", "content": prompt})

                # Chamamos o groq para um resposta
                completion = self.groq_client.chat.completions.create(
                    messages=self.feedback_history_report,
                    model=self.model,
                    temperature=temperature,
                    max_tokens=1000,
                )

                # Coloco então a resposta no histórico
                self.feedback_history_report.append({
                    "role": "assistant",
                    "content": completion.choices[0].message.content
                })

                # Retorno a resposta  
                return completion.choices[0].message.content

        # Checamos uma exceção    
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return None
    
    # Pegamos uma ação com base na política atual do codeReviewer
    def act(self):
        action = self.policy.get_action(self.state)
        self.current_action = action
        self.actions[action]()
    
    # Função para gerar as notas pro code
    def review_code(self):
        # Formatamos um prompt com o código atual
        prompt = prompt_review_code.format(code = self.code)

        # Pegamos então uma resposta estruturada pro code
        response = self._get_llm_response(prompt, temperature=0.1, review_code_bool=True)

        # Queremos fazer algo apenas se a resposta for diferente de None
        if response != None:
            # Pegamos a tupla de notas e adicionamos no grades
            grades = parse_tuple(response)
            self.grades["grades_llm"] = grades

    # Gera um relatório analítico com base no código fornecido e armazena na variável self.report. 
    def create_report(self):
        
        # Formatamos um prompt estruturado com as grades
        prompt = prompt_create_report.format(code = self.code, ruff_metrics = self.grades["ruff"], mypy_metrics = self.grades["mypy"], bandit_metrics = self.grades["bandit"])

        # # Pedimos uma resposta do llm
        structured_feedback = self._get_llm_response(prompt)

        # Queremos atualizar apenas se não for None
        if structured_feedback != None:
            # Armazenar o relatório
            self.report = structured_feedback
        
    # Analizamos o código com ruff
    def _analyze_with_ruff(self) -> int:
        # Criamos um arquivo temporário
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as temp_file:
            temp_file.write(self.code)
            temp_file.flush()
            
            # Executa uma ação externa
            result = subprocess.run(
                [SCRIPTS_PATH + "ruff", temp_file.name],
                capture_output=True,
                text=True
            )
            
        # Calculamos a pontuação com base na quantidade de problemas encontrados
        issues = result.stdout.strip().count("\n") + 1 if result.stdout else 0
        # Reduzimos do score
        score = -issues
        # retornamos o score 
        return score

    def _analyze_with_mypy(self) -> int:
        # Criamos um arquivo temporário
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as temp_file:
            temp_file.write(self.code)
            temp_file.flush()

            # Executa uma ação externa
            result = subprocess.run(
                [SCRIPTS_PATH + "mypy", temp_file.name],
                capture_output=True,
                text=True
            )
            
        # Calcula a pontuação com base na quantidade de erros de tipo
        issues = result.stdout.strip().count("\n") if result.stdout else 0
        # Reduzimos do score
        score = -2*issues
        return score

    def _analyze_with_bandit(self) -> int:
        # Criamos um arquivo temporário
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as temp_file:
            temp_file.write(self.code)
            temp_file.flush()
            
            # Executa uma ação externa
            result = subprocess.run(
                [SCRIPTS_PATH + "bandit", "-r", temp_file.name],
                capture_output=True,
                text=True
            )
        
        # Calcula a pontuação com base na gravidade dos problemas
        severity_pattern = re.compile(r'Severity: (\w+)')
        severities = severity_pattern.findall(result.stdout)
        
        # Começamos e diminuimos de acordo com a gravidade
        score = 0
        for severity in severities:
            if severity == 'High':
                score -= 8
            elif severity == 'Medium':
                score -= 5
            elif severity == 'Low':
                score -= 3

        return score

    def static_analysis(self):
        # Executamos as ações acima
        ruff_score = self._analyze_with_ruff()
        mypy_score = self._analyze_with_mypy()
        bandit_score = self._analyze_with_bandit()

        # Atualizamos cada uma dessas notas
        self.grades["ruff_score"] = ruff_score
        self.grades["mypy_score"] = mypy_score
        self.grades["bandit_score"] = bandit_score

    # Executamos e avaliamos o code
    def execute_and_score_code(self) -> int:
        # Criamos um arquivo temporário
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as temp_file:
            temp_file.write(self.code)
            temp_file.flush()
            
            try:
                # Tentamos executar o código
                exec(open(temp_file.name).read())
                # Se conseguir, pontua como 10
                execution_score = 10
            # Se der erro, penalizaremos
            except Exception as e:
                # Erros críticos
                if isinstance(e, (SyntaxError, NameError, TypeError, AttributeError)):
                    execution_score = -30
                # Erros moderados  
                elif isinstance(e, (IndexError, KeyError, ValueError)):
                    execution_score = -20  
                # Erros menos graves
                else:
                    execution_score = -10  
        
        self.grades["execution_score"] = execution_score

    # Atualizamos a política do CodeReviewer
    def update_policy(self, state: Tuple, action: int, reward: float, next_state: Tuple) -> None:
        """
        Parâmetros
        state (Tuple):
        Representa o estado atual do sistema. É um vetor com 14 notas fornecidas pelo "judger" (um avaliador ou sistema de análise).
        Essas notas indicam a avaliação do relatório de código antes da aplicação de uma ação.

        action (int):
        A ação escolhida pelo modelo.
        Cada ação representa uma modificação ou decisão específica que o modelo pode tomar para melhorar o relatório ou atender a critérios de avaliação.

        reward (float):
        A recompensa obtida após a execução da ação no estado atual.
        Representa a soma das notas obtidas no novo relatório gerado após a aplicação da ação. É um indicador da eficácia da ação tomada.

        next_state (Tuple):
        O estado resultante após a execução da ação. É um novo vetor de 14 notas avaliando o relatório atualizado.
        """
        self.policy.update(state, action, reward, next_state)