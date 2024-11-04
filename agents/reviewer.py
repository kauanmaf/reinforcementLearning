from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
import subprocess
from policy import EpsilonGreedyPolicy
from dotenv import load_dotenv
import os
import re
import subprocess
import tempfile

load_dotenv()

# Access the environment variables

SCRIPTS_PATH = os.getenv("SCRIPTS_PATH")

class ReviewAction(Enum):
    STATIC_ANALYSIS = "static_analysis"
    EXECUTE_CODE = "execute_code"
    PROPOSE_REFACTORING = "propose_refactoring"
    IMPROVE_REPORT = "improve_report"

@dataclass
class CodeReviewResult:
    action: ReviewAction
    feedback: str
    score: float
    suggestions: List[str]

class CodeReviewer:
    def __init__(self, client: str, model: str = "llama3-8b-8192"):
        """
        Inicializamos o Code reviewer com o groq
        """
        self.groq_client = client
        self.model = model
        self.feedback_history = [{
                    "role": "system",
                    "content": "You are an expert code reviewer. Provide specific, actionable feedback."
                }]
        # Começamos a política de epsilon greedy
        self.policy = EpsilonGreedyPolicy(n_actions=len(ReviewAction))
        self.actions = [self.create_report, self.execute_and_score_code, self.review_code, self.static_analysis]
        
        self.code = "print('Hello World')"
        self.metrics = {"ruff": 0, "mypy": 0, "bandit" : 0}
        self.report = None
        self.current_state = None
        self.grades = {}

    def _get_llm_response(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Função que pede uma resposta ao llm.
        """
        try:
            # Merge feedback history and the new user message
            messages = self.feedback_history + [{"role": "user", "content": prompt}]

            # Call the Groq client
            completion = self.groq_client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=temperature,
                max_tokens=1000
            )

            # Update feedback history with the user message and assistant response
            self.feedback_history.append({"role": "user", "content": prompt})
            self.feedback_history.append({
                "role": "assistant",
                "content": completion.choices[0].message.content
            })

            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return "Unable to get LLM feedback at this time."

    def review_code(self):
        """
        Perform a structured code review using Groq with feedback in 10 numbered metrics.
        """
        
            
        # Manually create the structured prompt for Groq
        prompt = f"""Please review the following code based on these 10 metrics, providing feedback for each:
        
1. Functional correctness
2. Code readability and style
3. Documentation quality and clarity
4. Efficiency and performance optimization
5. Error handling and robustness
6. Code organization and modularity
7. Typing and static analysis
8. Security practices
9. Test coverage and quality
10. Suggested improvements and best practices

### Code:
```python
{self.code}
```

YOUR OUTPUT SHOULD BE A TUPLE WITH 10 GRADES LIKE THIS (int,int,int,int,int,int,int,int,int,int). JUST THAT!!!.
"""

        # Obtain structured feedback from Groq
        self.grades["grades_llm"] = self._get_llm_response(prompt, temperature=0.1)

        print(self.grades["grades_llm"])
    
    def create_report(self, info):
        """
        Gera um relatório com base no código fornecido e armazena na variável self.report.
        """
        code = info.get("code", "")
        
        # Construir prompt estruturado para o LLM
        prompt = f"""Por favor, gere um relatório de revisão para o seguinte código:

### Código:
{code}

### Critérios de Avaliação:
1. Correção do código (avaliar bugs e erros)
2. Qualidade de estilo (baseada nas métricas de ruff: {self.metrics["ruff"]})
3. Precisão de tipagem (baseada nas métricas de mypy: {self.metrics["mypy"]})
4. Melhoria de desempenho e eficiência
5. Sugestões de otimização e melhores práticas

"""

        # Obter resposta do LLM
        structured_feedback = self._get_llm_response(prompt)

        # Armazenar o relatório
        self.report = {
            "feedback": structured_feedback
        }

    def _analyze_with_ruff(self) -> int:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as temp_file:
            temp_file.write(self.code)
            temp_file.flush()
            
            result = subprocess.run(
                [SCRIPTS_PATH + "ruff", temp_file.name],
                capture_output=True,
                text=True
            )
            
        # Calcula a pontuação com base na quantidade de problemas encontrados
        issues = result.stdout.strip().count("\n") + 1 if result.stdout else 0
        score = -issues  # Cada problema reduz 1 ponto até o mínimo de 0
        return score

    def _analyze_with_mypy(self) -> int:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as temp_file:
            temp_file.write(self.code)
            temp_file.flush()
            
            result = subprocess.run(
                [SCRIPTS_PATH + "mypy", temp_file.name],
                capture_output=True,
                text=True
            )
            
        # Calcula a pontuação com base na quantidade de erros de tipo
        issues = result.stdout.strip().count("\n") if result.stdout else 0
        score = -2*issues  # Cada problema reduz 2 pontos até o mínimo de 0
        return score

    def _analyze_with_bandit(self) -> int:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as temp_file:
            temp_file.write(self.code)
            temp_file.flush()
            
            result = subprocess.run(
                [SCRIPTS_PATH + "bandit", "-r", temp_file.name],
                capture_output=True,
                text=True
            )
        
        # Calcula a pontuação com base na severidade dos problemas de segurança
        severity_pattern = re.compile(r'Severity: (\w+)')
        severities = severity_pattern.findall(result.stdout)
        
        score = 0  # Começa com uma pontuação máxima de 10
        for severity in severities:
            if severity == 'High':
                score -= 8
            elif severity == 'Medium':
                score -= 5
            elif severity == 'Low':
                score -= 3

        return score

    def static_analysis(self, info):
        # Executa as análises de qualidade
        ruff_score = self._analyze_with_ruff(self.code)
        mypy_score = self._analyze_with_mypy(self.code)
        bandit_score = self._analyze_with_bandit(self.code)

        self.grades["ruff_score"] = ruff_score
        self.grades["mypy_score"] = mypy_score
        self.grades["bandit_score"] = bandit_score

    def execute_and_score_code(self) -> int:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as temp_file:
            temp_file.write(self.code)
            temp_file.flush()
            
            # Tenta executar o código e captura exceções
            try:
                exec(open(temp_file.name).read())
                execution_score = 10  # Nenhum erro, pontuação máxima
            except Exception as e:
                # Reduz a pontuação dependendo do tipo de exceção
                if isinstance(e, (SyntaxError, NameError, TypeError, AttributeError)):
                    execution_score = -30  # Erros críticos
                elif isinstance(e, (IndexError, KeyError, ValueError)):
                    execution_score = -20  # Erros moderados
                else:
                    execution_score = -10  # Erros menos graves
        
        self.grades["execution_score"] = execution_score

    def get_policy_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current policy
        """
        return {
            "num_states": len(self.policy.q_table),
            "epsilon": self.policy.epsilon,
            "average_q_value": np.mean([np.mean(q_values) for q_values in self.policy.q_table.values()]),
            "max_q_value": np.max([np.max(q_values) for q_values in self.policy.q_table.values()]),
            "most_visited_state": max(self.policy.q_table.items(), key=lambda x: np.sum(x[1]))[0]
        }

    def update_policy(self, state: Tuple, action: int, reward: float, next_state: Tuple) -> None:
        """
        Update RL policy based on action results
        """
        self.policy.update(state, action, reward, next_state)

    def act(self, state):
        action = self.policy.get_action(state)
        answer = self.actions[action]()
        
        return answer