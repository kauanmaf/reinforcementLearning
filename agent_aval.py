from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
import ast
import numpy as np
import random
from groq import Groq
import subprocess
from policy import EpsilonGreedyPolicy

class ReviewAction(Enum):
    STATIC_ANALYSIS = "static_analysis"
    EXECUTE_CODE = "execute_code"
    PROPOSE_REFACTORING = "propose_refactoring"
    APPROVE_CODE = "approve_code"
    IMPROVE_REPORT = "improve_report"

@dataclass
class CodeReviewResult:
    action: ReviewAction
    feedback: str
    score: float
    suggestions: List[str]

class CodeReviewer:
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        """
        Inicializamos o Code reviewer com o groq
        """
        self.groq_client = Groq(api_key=api_key)
        self.model = model
        self.feedback_history = [{
                    "role": "system",
                    "content": "You are an expert code reviewer. Provide specific, actionable feedback."
                }]
        self.report = None
        
        # Começamos a política de epsilon greedy
        self.policy = EpsilonGreedyPolicy(n_actions=len(ReviewAction))
        self.current_state = (0, 0)
        self.metrics = {"ruff": 0, "mypy": 0, "bandit" : 0}

    def _get_llm_response(self, prompt: str) -> str:
        """
        Função que pede uma resposta ao llm.
        """
        try:
            completion = self.groq_client.chat.completions.create(
                messages=[self.feedback_history, {
                    "role": "user",
                    "content": prompt
                }],
                model=self.model,
                temperature=0.7,
                max_tokens=1000
            )
            self.feedback_history.append({
                    "role": "user",
                    "content": prompt
                })
            self.feedback_history.append({
                "role": "assistant",
                "content": completion.choices[0].message.content
            })
            
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return "Unable to get LLM feedback at this time."

    def static_analysis(self, info):
        # Rodar ruff para analisar estilo e linting
        self.ruff_metrics = self.run_ruff_analysis(info["code"])
        # Rodar mypy para verificar tipagem estática
        self.mypy_metrics = self.run_mypy_analysis(info["code"])
    
    def review_code(self, info: Dict[str, Any]) -> CodeReviewResult:
        
        return result

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
            "ruff_metrics": ruff_metrics,
            "mypy_metrics": mypy_metrics,
            "feedback": structured_feedback
        }

    def _create_llm_prompt(self, action: ReviewAction, code: str, report: str, base_result: CodeReviewResult) -> str:
        """
        Create context-aware prompt for LLM
        """
        prompt = f"""
        As an expert code reviewer, analyze the following code and provide specific feedback.
        
        Action being taken: {action.value}
        
        Code:
        ```python
        {code}
        ```
        
        Current Analysis:
        {base_result.feedback}
        
        Focus on actionable feedback that will help improve the code.
        """
        return prompt

    def update_policy(self, state: Tuple, action: int, reward: float, next_state: Tuple) -> None:
        """
        Update RL policy based on action results
        """
        self.policy.update(state, action, reward, next_state)
        
    def optimize_prompt(self, info: Dict[str, Any]) -> str:
        """
        Create optimized prompt based on current state and history
        """
        current_result = self.review_code(info)
        
        # Analyze feedback history for patterns
        common_issues = self._analyze_feedback_history()
        
        prompt = f"""
        Code Review Feedback:
        Action Taken: {current_result.action.value}
        
        Overall Score: {current_result.score:.2f}
        
        Feedback:
        {current_result.feedback}
        
        Common Issues Identified:
        {chr(10).join(f'- {issue}' for issue in common_issues)}
        
        Suggestions for Improvement:
        {chr(10).join(f'- {suggestion}' for suggestion in current_result.suggestions)}
        
        Current Learning State:
        - Exploration rate (epsilon): {self.policy.epsilon:.3f}
        - Number of states explored: {len(self.policy.q_table)}
        """
        
        return prompt


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