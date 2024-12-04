from agents.coder import Coder
from agents.reviewer import CodeReviewer
from agents.judger import Judger
from dotenv import load_dotenv
import os
from groq import Groq
import pandas as pd

# Classe do ambiente
class Environment:
    def __init__(self, coder: Coder, reviewer: CodeReviewer, judger: Judger, threshold = 130):
        # Instância do codador
        self.coder = coder
        # Instância do revisor
        self.reviewer = reviewer
        # Instância do julgador
        self.judger = judger
        # Variável para controlar o fim de um episódio
        self.done = False
        # Limiar de pontuação para encerramento de um episódio
        self.threshold = threshold
        # Número de passos em um episódio
        self.step_count = 0
        # Número de iterações
        self.num_iter = 0
    
    # Função para resetar o ambiente após um episódio
    def reset(self):
        self.coder.reset()
        self.reviewer.reset()
        self.done = False

    # Função para chamar o codador para gerar um novo código
    def _coder_gen_new_code(self):
        # Obtendo o novo estado do codador com base nas notas do revisor
        next_state = self.reviewer.get_coder_state_from_grades()
        # Recompensa como a soma das notas
        score = sum(next_state)
        # Atualizando a política do codador
        self.coder.update_policy(self.coder.state, self.coder.current_action, score, next_state)
        # Fazendo o codador agir
        self.coder.act()
        # Salva o novo código também no revisor
        self.reviewer.code = self.coder.code
        # Atualiza o estado do codador
        self.coder.state = next_state

        print("Coder: ", self.coder, score)

    # Função para chamar o julgador para avaliar o relatório
    def _get_judger_to_analize_report(self, last_score):
        # Obtendo o novo estado do revisor com base nas notas do julgador
        next_state = self.judger.judge(self.reviewer.report)

        # Se der erro, retorna o estado e a recompensa anterior
        if next_state == None:
            return self.reviewer.state, last_score
        
        # Obtendo a recompensa como a soma das notas
        score = sum(next_state)

        # Se a recompensa (a menos da quantidade de passos) tiver ultrapassado o limiar, encerra o episódio
        if score + self.step_count > self.threshold:
            self.done = True

        return next_state, score

    def run_episode(self):
        # Máximo de passos por episódio
        max_steps = 50
        self.step_count = 0
        score = 0
        iteration_data = []

        # Gerando o primeiro código
        self.coder.act()
        self.reviewer.code = self.coder.code

        while not self.done:
            current_reviewer_state = self.reviewer.state
            self.reviewer.act()

            if self.reviewer.current_action == 2:  # "review code"
                self._coder_gen_new_code()
                next_state = self.reviewer.state
                score = sum(next_state) - 3 * self.step_count

            elif self.reviewer.current_action == 0:  # "create report"
                next_state, score = self._get_judger_to_analize_report(score)
                score -= 3 * self.step_count

            else:  # Outra ação
                next_state = self.reviewer.state
                score = sum(next_state) - 3 * self.step_count

            # Atualizando políticas
            self.reviewer.update_policy(current_reviewer_state, self.reviewer.current_action, score, next_state)
            self.reviewer.state = next_state

            # Incrementando passos
            self.step_count += 1

            if self.step_count > max_steps:
                self.reviewer.update_policy(current_reviewer_state, self.reviewer.current_action, -50, next_state)
                break

            iteration_data.append({
                "Episódio": self.num_iter,
                "Iteração": self.step_count,
                "Ação Reviewer": self.reviewer.current_action,
                "Descrição da ação": self.reviewer.action_names[self.reviewer.current_action], 
                "Recompensa": score
            })

            print(f"Iteração {self.step_count} | Ação Reviewer: {self.reviewer.current_action} | "
                f"Recompensa: {score} | Estado Reviewer: {current_reviewer_state}")

        self.coder.policy.save(filepath="models/coder")
        self.reviewer.policy.save(filepath="models/reviewer")

        return score, iteration_data

    def save_iteration_data(self, data, filename):
        df = pd.DataFrame(data)
        if not os.path.exists(filename):
            df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, mode='a', index=False, header=False)

    def run(self):
        self.step_count = 1000
        self.num_iter = 0

        while self.step_count > 5 and self.num_iter < 15:
            print(f"\nIniciando Episódio: {self.num_iter + 1}")
            reward, iter_data = self.run_episode() 
            print(f"Recompensa Final do Episódio {self.num_iter + 1}: {reward}")
            self.save_iteration_data(iter_data, "resultado.csv")
            with open("results/report.txt", "w", encoding="utf-8") as f:
                f.write(self.reviewer.report)
            with open("results/code.txt", "w", encoding="utf-8") as f:
                f.write(self.coder.code)
            self.reset()
            self.num_iter += 1