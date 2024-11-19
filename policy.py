from typing import Dict, Tuple
import numpy as np
import random


class EpsilonGreedyPolicy:
    def __init__(self, n_actions: int, epsilon: float = 0.1, epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table: Dict[Tuple, np.ndarray] = {(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): np.array([0, 0, 0, 0])}
        
    def get_action(self, state: Tuple) -> int:
        """
        Choose action using epsilon-greedy strategy
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
            
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state: Tuple, action: int, reward: float, next_state: Tuple, alpha: float = 0.1, gamma: float = 0.9):
        """
        Update Q-values using Q-learning
        """
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_actions)
            
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        
        # Q-learning update
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        self.q_table[state][action] = new_value
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        print(self.q_table)