from agents.coder import Coder
from agents.reviewer import CodeReviewer
from agents.judger import Judger
from dotenv import load_dotenv
import os
from groq import Groq
import pandas as pd

class Environment:
    def __init__(self, coder: Coder, reviewer: CodeReviewer, judger: Judger, threshold = 135):
        self.coder = coder
        self.reviewer = reviewer
        self.judger = judger
        self.done = False
        self.threshold = threshold
        self.step_count = 0
    
    def reset(self):
        self.coder.reset()
        self.reviewer.reset()
        self.judger.reset()
        self.done = False

    def _coder_gen_new_code(self):
        next_state = self.reviewer.get_coder_state_from_grades()
        score = sum(next_state)
        self.coder.update_policy(self.coder.state, self.coder.current_action, score, next_state)
        self.coder.act()
        self.reviewer.code = self.coder.code
        self.coder.state = next_state
        print("Coder: ", self.coder.current_action, score)

    def _get_judger_to_analize_report(self):
        next_state = self.judger.judge(self.reviewer.report)
        score = sum(next_state)

        if score > self.threshold:
            self.done = True

        return next_state, score
    
    def run_episode(self):
        """
        Executa uma rodada de aprendizado entre o coder e o reviewer.
        """
        total_reward = 0
        max_steps = 100
        self.step_count = 0

        # Gerando o primeiro código
        self.coder.act()
        # Salvando ele no reviewer também
        self.reviewer.code = self.coder.code

        while not self.done and self.step_count < max_steps:
            # Salvando o estado anterior do reviewer
            current_reviewer_state = self.reviewer.state
            # Fazer com que o reviewer aja:
            self.reviewer.act()

            # Se a ação do reviewer tiver sido review code
            if self.reviewer.current_action == 2:
                # E fazemos com que o coder gere um novo código
                self._coder_gen_new_code()
                next_state = self.reviewer.state
                score = sum(next_state) - self.step_count
            elif self.reviewer.current_action == 0:
                next_state, score = self._get_judger_to_analize_report()
                score -= self.step_count
            else:
                next_state = self.reviewer.state
                score = sum(next_state) - self.step_count

            self.reviewer.update_policy(current_reviewer_state, self.reviewer.current_action, score, next_state)
            self.reviewer.state = next_state
            
            self.step_count += 1

            print("Reviewer: ", self.reviewer.current_action, score)

        return total_reward
    
    def teste(self):
        total_reward = 0
        max_steps = 100
        self.step_count = 0
        i = 0

        # Gerando o primeiro código
        self.coder.act()
        # Salvando ele no reviewer também
        self.reviewer.code = self.coder.code

        while not self.done and self.step_count < max_steps:
            # Salvando o estado anterior do reviewer
            current_reviewer_state = self.reviewer.state
            # Fazer com que o reviewer aja:
            self.reviewer.actions[i%4]()
            self.reviewer.current_action = i%4
            i += 1

            # Se a ação do reviewer tiver sido review code
            if self.reviewer.current_action == 2:
                # E fazemos com que o coder gere um novo código
                self._coder_gen_new_code()
                next_state = self.reviewer.state
                score = sum(next_state) - self.step_count
            elif self.reviewer.current_action == 0:
                next_state, score = self._get_judger_to_analize_report()
                score -= self.step_count
            else:
                next_state = self.reviewer.state
                score = sum(next_state) - self.step_count

            self.reviewer.update_policy(current_reviewer_state, self.reviewer.current_action, score, next_state)
            self.reviewer.state = next_state
            
            self.step_count += 1

            print("Reviewer: ", self.reviewer.current_action, score)

        return total_reward