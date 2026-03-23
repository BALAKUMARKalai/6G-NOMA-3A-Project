import numpy as np
#U1 Utilisateur Faible (Loin)
#U2 Utilisateur Fort (Proche)

class NOMA_Simulator:
    
    def __init__(self):
        self.P_max = 7.0 #10
        self.bruit = 0.05
        self.gain_moyen_U1 = 0.2 #0.2
        self.gain_moyen_U2 = 0.5 #0.8 
        self.P_circuit = 5.0
        self.R_target_1 = 1.0 #(utilisateur lointain)
        self.R_target_2 = 3.0 #(utilisateur proche)
        self.Gamma_1 = (2**self.R_target_1) - 1
        self.Gamma_2 = (2**self.R_target_2) - 1
        self.g1 = 0.0
        self.g2 = 0.0

    def generate_channels_gains(self):
        self.g1 = np.random.exponential(scale=self.gain_moyen_U1)
        self.g2 = np.random.exponential(scale=self.gain_moyen_U2)
        return self.g1, self.g2
    
    def _evaluate(self, alpha, ordre_sic=None):
        g1 = self.g1
        g2 = self.g2

    # Détermination de l'ordre SIC
        if ordre_sic is not None:
            sic = 1 if ordre_sic > 0.5 else 0
        else:
            sic = 1   # défaut : U2 fait SIC

        if sic == 1:
        # Hypothèse : U2 est fort → U2 fait le SIC
            g_sic  = g2
            g_weak = g1
            P_weak = alpha * self.P_max
            P_sic  = (1 - alpha) * self.P_max
        else:
        # Hypothèse : U1 est fort → U1 fait le SIC
            g_sic  = g1
            g_weak = g2
            P_weak = (1 - alpha) * self.P_max
            P_sic  = alpha * self.P_max

        # Calcul des SINR
        SINR_weak      = (P_weak * g_weak) / (P_sic * g_weak + self.bruit)
        SINR_sic_step1 = (P_weak * g_sic)  / (P_sic * g_sic  + self.bruit)
        SINR_sic_step2 = (P_sic  * g_sic)  / self.bruit

        success_weak = (SINR_weak      >= self.Gamma_1)
        success_sic  = (SINR_sic_step1 >= self.Gamma_1) and \
                   (SINR_sic_step2 >= self.Gamma_2)

        return 1.0 if (success_weak and success_sic) else 0.0
    
    def step(self, alpha_coordinate):
    # HOO passe maintenant un vecteur [alpha, ordre_sic]
        if isinstance(alpha_coordinate, (list, np.ndarray)):
            alpha     = float(np.clip(alpha_coordinate[0], 0.01, 0.99))
            ordre_sic = float(alpha_coordinate[1]) if len(alpha_coordinate) > 1 else None
        else:
            alpha     = float(np.clip(alpha_coordinate, 0.01, 0.99))
            ordre_sic = None

        self.generate_channels_gains()
        reward   = self._evaluate(alpha, ordre_sic)
        feedback = [1 if reward == 1.0 else 0]
        return reward, feedback

    def check_possibility(self, alpha_coordinate, g1, g2):
        self.g1 = g1
        self.g2 = g2
        if isinstance(alpha_coordinate, (list, np.ndarray)):
            alpha     = float(alpha_coordinate[0])
            ordre_sic = float(alpha_coordinate[1]) if len(alpha_coordinate) > 1 else None
        else:
            alpha     = float(alpha_coordinate)
            ordre_sic = None
        return self._evaluate(alpha, ordre_sic)
    
class NOMA_Adapter:
    def __init__(self):
        self.env = NOMA_Simulator()
        self.drawn_values = []
        self.bests = [] 
        self.max_theoretical_reward = 1.0 

    def get_reward(self, alpha_coordinate):
        # Passe le vecteur complet [alpha, ordre_sic] à step()
        reward, feedbacks = self.env.step(alpha_coordinate)
        self.drawn_values.append(reward)

        g1_curr = self.env.g1
        g2_curr = self.env.g2

        # Oracle : teste toutes les combinaisons (alpha, ordre_sic)
        best_outcome = 0.0
        for a_test in np.linspace(0.01, 0.99, 20):
            for sic_test in [0.0, 1.0]:
                r_test = self.env.check_possibility(
                [a_test, sic_test], g1_curr, g2_curr
            )
                if r_test == 1.0:
                    best_outcome = 1.0
                    break
            if best_outcome == 1.0:
                break

        self.bests.append(best_outcome)
        return reward