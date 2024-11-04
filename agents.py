import os
from groq import Groq
from policy import EpsilonGreedyPolicy
import numpy as np
import ast

class Coder():
    def __init__(self, client, problem):
        self.client = client
        self.problem = problem
        self.actions = [self.process_data, self.analyze_data, self.visualize_results, self.interpret_analysis]
        self.policy = EpsilonGreedyPolicy(4)
        self.history = [{"role": "system",
                         "content": f"You are a Python developer and data scientist. Your role is to write code to solve the following data science problem: {self.problem}. Be concise and make sure to document your code."}]
        

    def act(self, state):
        action = self.policy.get_action(state)
        answer = self.actions[action]()
        
        return answer
    

    def update_policy(self, state, action, reward, next_state):
        self.policy.update(state, action, reward, next_state)


    def process_data(self):
        message = "Do the following in the code: clean, transform, and prepare the data for analysis."

        self.history.append({"role": "user",
                             "content": message})
        
        answer = self.client.chat.completions.create(
            messages = self.history,
            model = "llama3-8b-8192"
        )

        self.history.append({"role": "assistant",
                             "content": answer})
        
        return answer
    

    def analyze_data(self):
        message = "Do the following in the code: perform statistical analyses or build machine learning models."

        self.history.append({"role": "user",
                             "content": message})
        
        answer = self.client.chat.completions.create(
            messages = self.history,
            model = "llama3-8b-8192"
        )

        self.history.append({"role": "assistant",
                             "content": answer})
        
        return answer
    

    def visualize_results(self):
        message = "Do the following in the code: generate charts and data visualizations."

        self.history.append({"role": "user",
                             "content": message})
        
        answer = self.client.chat.completions.create(
            messages = self.history,
            model = "llama3-8b-8192"
        )

        self.history.append({"role": "assistant",
                             "content": answer})
        
        return answer
    

    def interpret_analysis(self):
        message = "Do the following in the code: interpret the results, producing text that includes analysis results in the form of figures and tables."

        self.history.append({"role": "user",
                             "content": message})
        
        answer = self.client.chat.completions.create(
            messages = self.history,
            model = "llama3-8b-8192"
        )

        self.history.append({"role": "assistant",
                             "content": answer})
        
        return answer
    

class Judger():
    def __init__(self, client, problem):
        self.client = client
        self.problem = problem
        self.history = [{"role": "system",
                         "content": f"""You are a critical report evaluator. Your role is to assign scores in a structured way to the reports provided, which address the following problem: {self.problem}. Write ONLY (DO NOT WRITE ANYTHING ELSE BEYOND!!!) a tuple with scores from 0 to 10 for each of the following topics:

                         - Clarity: Is the problem description clear and understandable?
                         - Accuracy: Is the problem description accurate and relevant?
                         - Completeness: Are all relevant data sets described in sufficient detail?
                         - Data Quality Analysis: Is the quality and characteristics of the data discussed (e.g., missing values, outliers)?
                         - Visualization: Is the data well-visualized using charts or tables?
                         - Approach: Is the chosen methodology appropriate to solve the data analysis problem?
                         - Justification: Is there a clear justification for why specific methods or techniques were chosen?
                         - Implementation: Is the implementation of the methodology accurately and thoroughly described?
                         - Precision: Are the results accurate and consistent with the analysis objectives?
                         - Understanding: Are the results adequately interpreted and discussed?
                         - Visualization: Are the results effectively visualized and easy to understand (e.g., charts, tables)?
                         - Summary: Does the conclusion succinctly summarize the main findings of the report?
                         - Implications: Are the implications of the results discussed?
                         - Recommendations: Are there actionable recommendations or next steps?
                         """}]
        
    def judge(self, report):
        self.history.append({"role": "user",
                             "content": report})
        
        answer = self.client.chat.completions.create(
            messages = self.history,
            model = "llama3-8b-8192",
            temperature = 0
        )

        self.history.append({"role": "assistant",
                             "content": answer})
        
        answer = ast.literal_eval(answer.choices[0].message.content)
        
        return answer
    

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

teste = Judger(client, "Test")
answer = teste.judge("Test")
print(answer)
