import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        :param input_size: 2 (Action + Reward)
        :param hidden_size: 64 (Taille de la mémoire du LSTM)
        :param output_size: K_actions (Nombre d'actions discrètes, ex: 10)
        """
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # LSTM avec batch_first=True
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            batch_first=True
        )
        
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state=None):
        """
        :param x: (Batch_Size, Sequence_Length, Input_Size)
        :param hidden_state: Tuple (h_0, c_0)
        :return: q_values (Batch, Seq_Len, Output_Size), hidden_state
        """
        
        # 1. Feature Extraction (Appliqué sur chaque pas de temps)
        # x : (Batch, Seq, Hidden)
        x = F.relu(self.fc1(x))
        
        # 2. LSTM
        # lstm_out : (Batch, Seq, Hidden) -> Contient TOUTE la séquence
        lstm_out, new_hidden_state = self.lstm(x, hidden_state)
        
        # 3. Calcul des Q-Values
        # --- CORRECTION CRUCIALE ICI ---
        # On ne prend plus juste le dernier élément [-1].
        # On passe toute la séquence 'lstm_out' dans la couche linéaire.
        # PyTorch va transformer (Batch, Seq, Hidden) -> (Batch, Seq, Output)
        q_values = self.fc2(lstm_out)
        
        return q_values, new_hidden_state