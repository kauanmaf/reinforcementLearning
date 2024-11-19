import os
import sys
from parser import *

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
        self.history = [{"role": "system",
                         "content": prompt_judger.format(problem = self.problem)},
                        {"role": "user",
                         "content": report}]

        answer = None

        while not answer:
            answer = self.client.chat.completions.create(
                messages = self.history,
                model = "gemma-7b-it",
                temperature = 0,
                max_tokens = 65
            )

            answer = answer.choices[0].message.content
            print(answer)
            answer = parse_tuple(answer)
        
        return answer