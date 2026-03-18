
import numpy as np

"""
Environnement NOMA Gauss-Markov pour RDPG.
Action continue : [alpha, ordre_sic_logit] ∈ [0,1]²
"""

class GaussMarkov:
    def __init__(self, bruit=0.01):
        self.P_max         = 7.0
        self.bruit         = bruit
        self.P_circuit     = 0.1
        self.gain_moyen_U1 = 0.2
        self.gain_moyen_U2 = 0.5
        self.R_target_1    = 1.0
        self.R_target_2    = 2.0
        self.Gamma_1       = (2 ** self.R_target_1) - 1
        self.Gamma_2       = (2 ** self.R_target_2) - 1
        self.rho           = 0.5
        self.h1_curr       = None
        self.h2_curr       = None

        self.K_alpha   = 20
        self.K_actions = self.K_alpha * 2
    def _init_complex_channel(self, avg_gain):
        std = np.sqrt(avg_gain / 2.0)
        return np.random.normal(0, std) + 1j * np.random.normal(0, std)

    def reset(self):
        self.h1_curr = self._init_complex_channel(self.gain_moyen_U1)
        self.h2_curr = self._init_complex_channel(self.gain_moyen_U2)
        return [0.0, 0.0]

    def _evaluate(self, alpha, ordre_sic):
        g1 = np.abs(self.h1_curr) ** 2
        g2 = np.abs(self.h2_curr) ** 2

        if ordre_sic == 1:
            g_sic  = g2
            g_weak = g1
            P_weak = alpha * self.P_max
            P_sic  = (1 - alpha) * self.P_max
        else:
            g_sic  = g1
            g_weak = g2
            P_weak = (1 - alpha) * self.P_max
            P_sic  = alpha * self.P_max

        SINR_weak      = (P_weak * g_weak) / (P_sic * g_weak + self.bruit)
        SINR_sic_step1 = (P_weak * g_sic)  / (P_sic * g_sic  + self.bruit)
        SINR_sic_step2 = (P_sic  * g_sic)  / self.bruit

        success_weak = (SINR_weak      >= self.Gamma_1)
        success_sic  = (SINR_sic_step1 >= self.Gamma_1) and (SINR_sic_step2 >= self.Gamma_2)

        return 1.0 if (success_weak and success_sic) else 0.0

    def step(self, action):
        """
        action : vecteur numpy [alpha, ordre_sic_logit] ∈ [0,1]²
        Retourne (next_obs, reward, done)
        """
        alpha     = float(np.clip(action[0], 0.01, 0.99))
        ordre_sic = 1 if action[1] > 0.5 else 0

        reward = self._evaluate(alpha, ordre_sic)

        noise1 = self._init_complex_channel(self.gain_moyen_U1)
        noise2 = self._init_complex_channel(self.gain_moyen_U2)
        self.h1_curr = self.rho * self.h1_curr + np.sqrt(1 - self.rho ** 2) * noise1
        self.h2_curr = self.rho * self.h2_curr + np.sqrt(1 - self.rho ** 2) * noise2

        next_obs = [float(alpha), float(reward)]
        return next_obs, reward, False

    def _decode_action(self, action_index):
        """Utilisé uniquement par l'oracle."""
        ordre_sic   = 0 if action_index < self.K_alpha else 1
        alpha_index = action_index % self.K_alpha
        alpha       = (alpha_index + 1) / (self.K_alpha + 1.0)
        return alpha, ordre_sic

    def get_oracle_reward(self):
        """
        Oracle : teste 40 combinaisons discrètes (alpha, ordre_sic)
        sur les canaux courants. Appelé APRÈS step().
        """
        return max(
            self._evaluate(*self._decode_action(a))
            for a in range(self.K_actions)
        )