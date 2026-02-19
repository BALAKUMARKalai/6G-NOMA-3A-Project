import numpy as np
# U1 : Utilisateur Faible (Loin)
# U2 : Utilisateur Fort (Proche)

class GaussMarkov:
    def __init__(self, bruit=0.01):
        self.P_max      = 10.0
        self.bruit      = bruit
        self.P_circuit  = 0.1

        self.gain_moyen_U1 = 0.2
        self.gain_moyen_U2 = 1.0

        self.R_target_1 = 0.5  
        self.R_target_2 = 1.0   

        self.Gamma_1 = (2 ** self.R_target_1) - 1
        self.Gamma_2 = (2 ** self.R_target_2) - 1

        self.rho = 0.95   

        self.h1_curr = None
        self.h2_curr = None

        self.K_actions = 20   #

    def _init_complex_channel(self, avg_gain):
        std = np.sqrt(avg_gain / 2.0)
        return np.random.normal(0, std) + 1j * np.random.normal(0, std)

    def reset(self):
        self.h1_curr = self._init_complex_channel(self.gain_moyen_U1)
        self.h2_curr = self._init_complex_channel(self.gain_moyen_U2)
        return [0.0, 0.0]

    def _evaluate(self, alpha):
        """
        Calcule le succès NOMA pour un alpha donné et les canaux courants.
        Retourne reward binaire (1 ou 0) — cohérent avec le feedback 1-bit du papier.
        """
        g1 = np.abs(self.h1_curr) ** 2
        g2 = np.abs(self.h2_curr) ** 2

        P1, P2 = alpha * self.P_max, (1 - alpha) * self.P_max

        SINR_1 = (P1 * g1) / (P2 * g1 + self.bruit)
        SINR_2_sic = (P1 * g2) / (P2 * g2 + self.bruit)
        SINR_2_own = (P2 * g2) / self.bruit
        success_U1 = (SINR_1     >= self.Gamma_1)
        success_U2 = (SINR_2_sic >= self.Gamma_1) and (SINR_2_own >= self.Gamma_2)

        return 1.0 if (success_U1 and success_U2) else 0.0

    def step(self, action_index):
        """
        Action : indice discret → alpha continu
        Observation : [alpha_t, feedback_1bit_t]
        Reward : 1 (succès) ou 0 (outage) — feedback 1-bit
        """
        alpha = (action_index + 1) / (self.K_actions + 1.0)
        reward = self._evaluate(alpha)
        noise1 = self._init_complex_channel(self.gain_moyen_U1)
        noise2 = self._init_complex_channel(self.gain_moyen_U2)
        self.h1_curr = self.rho * self.h1_curr + np.sqrt(1 - self.rho ** 2) * noise1
        self.h2_curr = self.rho * self.h2_curr + np.sqrt(1 - self.rho ** 2) * noise2
        next_obs = [float(alpha), float(reward)]
        done = False
        return next_obs, reward, done

    def get_oracle_reward(self):
        """
        Oracle : connaît les canaux courants, cherche le meilleur alpha.
        Appelé APRÈS step() pour utiliser les mêmes canaux.
        """
        return max(
            self._evaluate((a + 1) / (self.K_actions + 1.0))
            for a in range(self.K_actions)
        )