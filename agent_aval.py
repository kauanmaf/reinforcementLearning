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
        Initialize CodeReviewer with Groq client
        """
        self.groq_client = Groq(api_key=api_key)
        self.model = model
        self.feedback_history = []
        
        # Initialize RL policy
        self.policy = EpsilonGreedyPolicy(n_actions=len(ReviewAction))
        self.current_state = (0, 0)  # Initial state

    def _get_llm_response(self, prompt: str) -> str:
        """
        Get response from Groq LLM
        """
        try:
            completion = self.groq_client.chat.completions.create(
                messages=[{
                    "role": "system",
                    "content": "You are an expert code reviewer. Provide specific, actionable feedback."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                model=self.model,
                temperature=0.7,
                max_tokens=1000
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return "Unable to get LLM feedback at this time."

    def review_code(self, info: Dict[str, Any]) -> CodeReviewResult:
        """
        Review code using RL policy and LLM feedback
        """
        code = info.get("code", "")
        report = info.get("report", "")
        metrics = self._get_current_metrics(code, report)
        
        # Get state representation
        state = self._get_state_representation(metrics)
        
        # Select action using epsilon-greedy policy
        action_idx = self.policy.get_action(state)
        action = list(ReviewAction)[action_idx]
        
        # Execute selected action
        result = self._execute_action(action, code, report, metrics)
        
        # Update policy based on reward
        next_metrics = self._get_current_metrics(code, report)
        next_state = self._get_state_representation(next_metrics)
        self.update_policy(state, action_idx, result.score, next_state)
        
        # Store feedback
        self.feedback_history.append(result)
        
        return result

    def _execute_action(self, action: ReviewAction, code: str, report: str, metrics: Dict[str, float]) -> CodeReviewResult:
        """
        Execute selected action and get LLM feedback
        """
        base_result = super()._execute_action(action, code, report, metrics)
        
        # Enhance feedback with LLM
        prompt = self._create_llm_prompt(action, code, report, base_result)
        llm_feedback = self._get_llm_response(prompt)
        
        # Combine automated and LLM feedback
        enhanced_result = CodeReviewResult(
            action=base_result.action,
            feedback=f"{base_result.feedback}\n\nDetailed Analysis:\n{llm_feedback}",
            score=base_result.score,
            suggestions=base_result.suggestions + self._parse_llm_suggestions(llm_feedback)
        )
        
        return enhanced_result

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

    def _analyze_feedback_history(self) -> List[str]:
        """
        Analyze feedback history to identify common patterns
        """
        if not self.feedback_history:
            return []
            
        issue_counter: Dict[str, int] = {}
        for result in self.feedback_history:
            for suggestion in result.suggestions:
                issue_counter[suggestion] = issue_counter.get(suggestion, 0) + 1
                
        # Return most common issues
        return [issue for issue, count in sorted(
            issue_counter.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]]

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