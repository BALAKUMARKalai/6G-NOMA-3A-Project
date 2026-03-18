
import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    Critique récurrent (LSTM) pour RDPG.
    Entrée  : (observation, action) concaténés
    Sortie  : Q-value scalaire
    """
    def __init__(self, obs_size, hidden_size, action_size=2):
        super(Critic, self).__init__()
        self.fc1  = nn.Linear(obs_size + action_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc2  = nn.Linear(hidden_size, 1)

    def forward(self, obs, action, hidden=None):
        """
        obs    : (Batch, Seq, obs_size)
        action : (Batch, Seq, action_size)
        """
        x = torch.cat([obs, action], dim=-1)   # (Batch, Seq, obs+action)
        x = F.relu(self.fc1(x))
        x, hidden = self.lstm(x, hidden)
        q_value = self.fc2(x)                  # (Batch, Seq, 1)
        return q_value, hidden