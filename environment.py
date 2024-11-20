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

    # Função para rodar um episódio
    def run_episode(self):
        # Máximo de passos por episódio
        max_steps = 50
        # Reiniciando a contagem de passos
        self.step_count = 0
        # score
        score = 0

        # Gerando o primeiro código
        self.coder.act()
        # Salvando ele no reviewer também
        self.reviewer.code = self.coder.code

        # Enquanto não tiver concluído o episódio...
        while not self.done:
            # Salva o estado anterior do reviewer
            current_reviewer_state = self.reviewer.state
            # Faz com que o revisor aja
            self.reviewer.act()

            # Se a ação do revisor tiver sido "review code"...
            if self.reviewer.current_action == 2:
                # Faz com que o codador gere um novo código
                self._coder_gen_new_code()
                # Atualiza o estado do revisor
                next_state = self.reviewer.state
                # Salva a recompensa como a soma das notas retirada o número de passos até agora ponderado
                score = sum(next_state) - 3*self.step_count
            # Se a ação do revisor tiver sido "create report"...
            elif self.reviewer.current_action == 0:
                # Faz com que o julgador avalie o novo relatório
                next_state, score = self._get_judger_to_analize_report(score)
                # Diminui o número de passos da pontuação ponderado
                score -= 3*self.step_count
            # Se não...
            else:
                # O próximo estado do revisor é seu estado atual
                next_state = self.reviewer.state
                # Sua recompensa é a soma das notas atuais menos o número de passos até agora ponderado
                score = sum(next_state) - 3*self.step_count

            # Atualiza a política do revisor
            self.reviewer.update_policy(current_reviewer_state, self.reviewer.current_action, score, next_state)
            # Atualiza o estado do revisor
            self.reviewer.state = next_state
            
            # Incrementa o número de passos
            self.step_count += 1

            # Se o número de passos tiver passado do limite...
            if self.step_count > max_steps:
                # Atualiza a política do revisor punindo-o por não ter concluído a tarefa
                self.reviewer.update_policy(current_reviewer_state, self.reviewer.current_action, -50, next_state)
                # Finaliza o episódio
                break

            print("Iteração: ", self.step_count)
            print("Reviewer: ", self.reviewer, score)

        # Salvando os modelos
        self.coder.policy.save(filepath="models/coder")
        self.reviewer.policy.save(filepath="models/reviewer")

        return score
    
    # Função para rodar o modelo completo
    def run(self):

        # Inicializando o número de passos de um episódio
        self.step_count = 1000
        # Número de episódios
        num_iter = 0
        # Dicionário para armazenar os resultados
        results = {"Episódio": [], "Iterações": [], "Score final": []}

        # Enquanto o número de passos for maior que 5 e o número de episódios for menor que 15
        while self.step_count > 5 and num_iter < 15:
            print(f"Episódio: {num_iter + 1}")

            # Executa o episódio e obtém o score final
            reward = self.run_episode() 
            print(f"Reward: {reward}")

            # Adiciona os resultados ao dicionário
            results["Episódio"] = [num_iter + 1]
            results["Iterações"] = [self.step_count]
            results["Score final"] = [reward]

            # Converte os resultados para um DataFrame
            df = pd.DataFrame(results)

            # Salva os resultados em um arquivo CSV
            if not os.path.exists("resultados_episodios.csv"):
                df.to_csv("resultados_episodios.csv", index=False)
            else:
                df.to_csv("resultados_episodios.csv", mode='a', index=False, header=False)

            # Reinicia os estados
            self.reset()  
            # Incrementa o número de episódios
            num_iter += 1
            