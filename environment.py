from agents.coder import Coder
from agents.reviewer import CodeReviewer
from agents.judger import Judger
from dotenv import load_dotenv
import os
from groq import Groq
import pandas as pd

class Environment:
    def __init__(self, coder: Coder, reviewer: CodeReviewer, judger: Judger, threshold = 130):
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
        
        if next_state == None:
            return self.reviewer.state, score
        
        score = sum(next_state)

        if score + self.step_count > self.threshold:
            self.done = True

        return next_state, score

    def run_episode(self):
        """
        Executa uma rodada de aprendizado entre o coder e o reviewer.
        """
        max_steps = 50
        self.step_count = 0

        # Gerando o primeiro código
        self.coder.act()
        # Salvando ele no reviewer também
        self.reviewer.code = self.coder.code

        while not self.done:
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

            if self.step_count > max_steps:
                self.reviewer.update_policy(current_reviewer_state, self.reviewer.current_action, score - 50, next_state)
                break

            print("Iteração: ", self.step_count)
            print("Reviewer: ", self.reviewer.current_action, score)

        self.coder.policy.save(filepath="models/coder")
        self.reviewer.policy.save(filepath="models/reviewer")
        return score
    
    def run(self):
        self.step_count = 1000  # Apenas um número alto
        num_iter = 0
        results = {"Episódio": [], "Iterações": [], "Score final": []}  # Dicionário para armazenar os resultados

        while self.step_count > 5 or num_iter < 15:
            print("Episódio: ", num_iter + 1)
            reward = self.run_episode()  # Executa o episódio e obtém o score final
            print("Reward:", reward)
            
            # Adiciona os resultados ao dicionário
            results["Episódio"].append(num_iter + 1)
            results["Iterações"].append(self.step_count)
            results["Score final"].append(reward)
            
            self.reset()  # Reinicia os estados
            num_iter += 1
            # Cria um DataFrame com os resultados
            df = pd.DataFrame(results)
            # Salva os resultados em um arquivo CSV
            df.to_csv("resultados_episodios.csv", index=False)
            