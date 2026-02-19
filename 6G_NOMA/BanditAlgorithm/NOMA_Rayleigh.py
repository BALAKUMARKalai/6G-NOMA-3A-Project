import numpy as np
#U1 Utilisateur Faible (Loin)
#U2 Utilisateur Fort (Proche)

class NOMA_Simulator:
    
    def __init__(self):
        self.P_max = 5.0 #10
        self.bruit = 0.1
        self.gain_moyen_U1 = 0.05 #0.2
        self.gain_moyen_U2 = 1.5 #0.8 
        self.P_circuit = 5.0
        self.R_target_1 = 1.0 #(utilisateur lointain)
        self.R_target_2 = 2.0 #(utilisateur proche)
        self.Gamma_1 = (2**self.R_target_1) - 1
        self.Gamma_2 = (2**self.R_target_2) - 1
        self.g1 = 0.0
        self.g2 = 0.0

    def generate_channels_gains(self):
        self.g1 = np.random.exponential(scale=self.gain_moyen_U1)
        self.g2 = np.random.exponential(scale=self.gain_moyen_U2)
        return self.g1, self.g2
    
    def _evaluate(self, alpha, g1, g2):
        P1, P2 = alpha * self.P_max, (1 - alpha) * self.P_max
        SINR_1     = (P1 * g1) / (P2 * g1 + self.bruit)
        SINR_2_sic = (P1 * g2) / (P2 * g2 + self.bruit)
        SINR_2_own = (P2 * g2) / self.bruit

        success_U1 = (SINR_1>= self.Gamma_1)
        success_U2 = (SINR_2_sic >= self.Gamma_1) and (SINR_2_own >= self.Gamma_2)
    
    # Même reward pour tout le monde
        return 1.0 if (success_U1 and success_U2) else 0.0

    def step(self, alpha): #action de l'agent
        self.generate_channels_gains()
        reward = self._evaluate(alpha, self.g1, self.g2)
        feedback = [1 if reward == 1.0 else 0]
        return reward, feedback

    def check_possibility(self, alpha, g1, g2): #oracle
        return self._evaluate(alpha, g1, g2)  # exactement la même logique
    
class NOMA_Adapter:
    def __init__(self):
        self.env = NOMA_Simulator()
        self.drawn_values = []
        self.bests = [] 
        self.max_theoretical_reward = 1.0 

    def get_reward(self, alpha_coordinate):
        if isinstance(alpha_coordinate, (list, np.ndarray)):
            alpha = float(alpha_coordinate[0])
        else:
            alpha = float(alpha_coordinate)

        reward, feedbacks = self.env.step(alpha)
        self.drawn_values.append(reward)

        g1_curr = self.env.g1
        g2_curr = self.env.g2
        best_outcome = 0.0
        
        test_alphas = np.linspace(0.01, 0.99, 50) 
        
        for a_test in test_alphas:
            r_test = self.env.check_possibility(a_test, g1_curr, g2_curr)
            if r_test == 1.0:
                best_outcome = 1.0
                break 
        
        self.bests.append(best_outcome)
        return reward