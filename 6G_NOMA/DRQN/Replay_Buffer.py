import random
import numpy as np
import torch
from collections import deque #ajout et retrait d'élements dans une liste

class RecurrentReplayBuffer:
    def __init__(self, capacity, sequence_length):
        """
        :param capacity: Nombre maximum d'épisodes stockés (ex: 1000)
        :param sequence_length: Longueur de la trace temporelle pour le LSTM (ex: 10)
        """
        self.buffer = deque(maxlen=capacity)
        self.seq_len = sequence_length

    def push(self, episode):
        """
        Ajoute un épisode complet au buffer.Episode doit être une liste de tuples: [(obs, action, reward, next_obs, done), ...]
        """
        self.buffer.append(episode)

    def sample(self, batch_size):
        """
        Tire un batch de séquences aléatoires et Retourne des Tenseurs prêts pour l'entraînement.
        """
        b_obs, b_actions, b_rewards, b_next_obs, b_dones = [], [], [], [], []
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)

        for idx in indices:
            episode = self.buffer[idx]
            
            # Sécurité : Si l'épisode est trop court, on le saute
            if len(episode) < self.seq_len:
                start_index = 0
            else:
                # On choisit un point de départ aléatoire valide
                # pour avoir une séquence de taille 'seq_len'
                start_index = np.random.randint(0, len(episode) - self.seq_len + 1)
            trace = episode[start_index : start_index + self.seq_len]
            obs, action, reward, next_obs, done = zip(*trace) 

            b_obs.append(np.array(obs))
            b_actions.append(np.array(action))
            b_rewards.append(np.array(reward))
            b_next_obs.append(np.array(next_obs))
            b_dones.append(np.array(done))
        return {
            'obs':      torch.FloatTensor(np.array(b_obs)),
            'actions':  torch.LongTensor(np.array(b_actions)).unsqueeze(2),
            'rewards':  torch.FloatTensor(np.array(b_rewards)).unsqueeze(2),
            'next_obs': torch.FloatTensor(np.array(b_next_obs)),
            'dones':    torch.FloatTensor(np.array(b_dones)).unsqueeze(2)
        }

    def __len__(self):
        return len(self.buffer)
        