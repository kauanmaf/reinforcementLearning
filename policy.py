import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
import json
import random
import os


class EpsilonGreedyPolicyApprox:
    def __init__(self, state_dim: int, n_actions: int, epsilon: float = 0.2, epsilon_decay: float = 0.995, epsilon_min: float = 0.05, lr: float = 0.001, model_path=None):
        # Número de ações possíveis
        self.n_actions = n_actions
        # Taxa de exploração inicial
        self.epsilon = epsilon
        # Fator de decaimento do epsilon
        self.epsilon_decay = epsilon_decay
        # Limite inferior para o epsilon
        self.epsilon_min = epsilon_min

        # Define a rede neural para estimar Q(s, a)
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

        # Inicializa o otimizador e a função de perda
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Carrega modelo salvo, se disponível
        if model_path and os.path.exists(model_path):
            print(f"Loading policy model from {model_path}")
            self.model.load_state_dict(torch.load(model_path))
        else:
            print("No saved model found. Starting with a new model.")

    def get_action(self, state: Tuple) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Exploração
        else:
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()  # Exploração máxima

    def update(self, state: Tuple, action: int, reward: float, next_state: Tuple, alpha: float = 0.1, gamma: float = 0.9):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # Obtém o Q(s, a) atual
        q_values = self.model(state_tensor)
        q_value = q_values[0, action]

        # Calcula o valor alvo com base no próximo estado
        with torch.no_grad():
            next_q_values = self.model(next_state_tensor)
            next_max_q_value = next_q_values.max().item()

        target = torch.tensor([reward + gamma * next_max_q_value], dtype=torch.float32)

        # Calcula a perda e realiza backpropagation
        loss = self.loss_fn(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Atualiza o epsilon com decaimento
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        print(f"Updated Q-values: {q_values.detach().numpy()}")

    def save(self, filepath: str):
        # Salva o modelo e os parâmetros
        torch.save(self.model.state_dict(), filepath + "_model.pth")
        params = {
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
        }
        with open(filepath + "_params.json", "w") as f:
            json.dump(params, f)
        print(f"Modelo salvo em {filepath}_model.pth e parâmetros em {filepath}_params.json")

    def load(self, filepath: str):
        # Carrega o modelo e os parâmetros
        self.model.load_state_dict(torch.load(filepath + "_model.pth"))
        self.model.eval()
        with open(filepath + "_params.json", "r") as f:
            params = json.load(f)
        self.epsilon = params["epsilon"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon_min = params["epsilon_min"]
        print(f"Modelo carregado de {filepath}_model.pth e parâmetros de {filepath}_params.json")
