import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple
import random

class EpsilonGreedyPolicyApprox:
    def __init__(self, state_dim: int, n_actions: int, epsilon: float = 0.1, epsilon_decay: float = 0.995, epsilon_min: float = 0.01, lr: float = 0.001):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Rede neural para aproximar Q(s, a)
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        
        # Otimizador
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def get_action(self, state: Tuple) -> int:
        """
        Escolhe uma ação usando a estratégia epsilon-greedy
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Transforma o estado para um tensor
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def update(self, state: Tuple, action: int, reward: float, next_state: Tuple, alpha: float = 0.1, gamma: float = 0.9):
        """
        Atualiza os valores Q usando aproximação de função com uma rede neural
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # Obtenha o valor Q(s, a) atual
        q_values = self.model(state_tensor)
        q_value = q_values[0, action]

        # Calcula o valor alvo
        with torch.no_grad():
            next_q_values = self.model(next_state_tensor)
            next_max_q_value = next_q_values.max().item()
        target = reward + gamma * next_max_q_value

        # Calcula a perda e realiza o backpropagation
        loss = self.loss_fn(q_value, torch.tensor(target))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decai o epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        print(f"Updated Q-values: {q_values.detach().numpy()}")
