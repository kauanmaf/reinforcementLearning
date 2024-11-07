from agents.coder import Coder
from agents.reviewer import CodeReviewer
from agents.judger import Judger
from dotenv import load_dotenv
import os
from groq import Groq
import pandas as pd

class Environment:
    def __init__(self, coder: Coder, reviewer: CodeReviewer, judger: Judger, threshold = 0.9):
        self.coder = coder
        self.reviewer = reviewer
        self.judger = judger
        self.done = False
        self.threshold = threshold
    
    def reset(self):
        self.coder.reset()
        self.reviewer.reset()
        self.judger.reset()
        self.done = False

    # def _coder_gen_new_code(self):


    def _get_judger_to_analize_report(self, report):
        score = self.judger.judge(report)
        score = sum(score)
        if score > self.threshold:
            self.done = True
    
    def run_episode(self):
        """
        Executa uma rodada de aprendizado entre o coder e o reviewer.
        """
        total_reward = 0
        max_steps = 100
        step_count = 0

        while not self.done and step_count < max_steps:
            coder_action = self.coder.act(self.state)
            reviewer_action = self.reviewer.act(self.state)
            
            state, reward, self.done = self.step(coder_action, reviewer_action)
            total_reward += reward
            step_count += 1

            # Atualizar a política dos agentes com as recompensas obtidas
            self.coder.update_policy(state, coder_action, reward, self.state)
            self.reviewer.update_policy(state, reviewer_action, reward, self.state)

        return total_reward
