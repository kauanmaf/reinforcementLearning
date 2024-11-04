from policy import EpsilonGreedyPolicy

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