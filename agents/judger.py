import os
import sys
from agents.utils import *

if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    os.chdir(parent_dir)
    sys.path.append(parent_dir)

import ast

with open("prompts/judger.txt", "r") as file:
    prompt_judger = file.read()

class Judger():
    def __init__(self, client, problem):
        self.client = client
        self.problem = problem
        self.history = [{"role": "system",
                         "content": prompt_judger.format(problem = self.problem)}]
        
    def reset(self):
        self.history = [{"role": "system",
                         "content": prompt_judger}]

    def judge(self, report):
        self.history.append({"role": "user",
                             "content": report})

        answer = self.client.chat.completions.create(
            response_model = ReportJudger, 
            messages = self.history,
            model = "llama3-8b-8192",
            temperature = 0
        )

        # Transformando a answer em apenas um tupla
        answer = answer.scores
        
        self.history.append({"role": "assistant",
                             "content": str(answer)})
        
        return answer