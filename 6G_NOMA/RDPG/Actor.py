import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Acteur récurrent (LSTM) pour RDPG.
    Entrée  : historique d'observations [alpha_t, feedback_t]
    Sortie  : action continue [alpha, ordre_sic_logit] ∈ [0,1]²
    """
    def __init__(self, input_size, hidden_size, action_size=2):
        super(Actor, self).__init__()
        self.fc1  = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc2  = nn.Linear(hidden_size, action_size)

    def forward(self, x, hidden=None):
        """
        x      : (Batch, Seq, input_size)
        hidden : tuple (h, c) du LSTM
        """
        x = F.relu(self.fc1(x))
        x, hidden = self.lstm(x, hidden)
        action = torch.sigmoid(self.fc2(x))   # ∈ [0,1] pour alpha ET ordre_sic
        return action, hidden