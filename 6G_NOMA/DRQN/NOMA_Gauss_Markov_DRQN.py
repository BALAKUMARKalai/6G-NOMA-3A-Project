
import numpy as np
"""
 l'action est composée du choix de la puissance et du choix de l'utilisateur qui va SIC
"""
class GaussMarkov:
    def __init__(self, bruit=0.01):
        self.P_max      = 7.0
        self.bruit      = bruit
        self.P_circuit  = 0.1
        self.gain_moyen_U1 = 0.2
        self.gain_moyen_U2 = 0.5
        self.R_target_1 = 1.0
        self.R_target_2 = 2.0
        self.Gamma_1 = (2 ** self.R_target_1) - 1
        self.Gamma_2 = (2 ** self.R_target_2) - 1
        self.rho = 0.95
        self.h1_curr = None
        self.h2_curr = None

        # Action = (alpha_index, ordre_sic)
        # alpha_index ∈ [0, K_alpha-1] donc 20 valeurs
        # ordre_sic   ∈ {0, 1}         donc 2 valeurs
        # Total : 20 × 2 = 40 actions
        self.K_alpha   = 20
        self.K_actions = self.K_alpha * 2   # 40 actions au total

    def _init_complex_channel(self, avg_gain):
        std = np.sqrt(avg_gain / 2.0)
        return np.random.normal(0, std) + 1j * np.random.normal(0, std)

    def reset(self):
        self.h1_curr = self._init_complex_channel(self.gain_moyen_U1)
        self.h2_curr = self._init_complex_channel(self.gain_moyen_U2)
        return [0.0, 0.0]

    def _decode_action(self, action_index):
        """
        Décode un indice d'action en (alpha, ordre_sic).
        action_index 0  à K_alpha-1       → ordre_sic=0, alpha variable
        action_index K_alpha à 2*K_alpha-1 → ordre_sic=1, alpha variable
        """
        ordre_sic   = 0 if action_index < self.K_alpha else 1
        alpha_index = action_index % self.K_alpha
        alpha       = (alpha_index + 1) / (self.K_alpha + 1.0)
        return alpha, ordre_sic

    def _evaluate(self, alpha, ordre_sic):
        g1 = np.abs(self.h1_curr) ** 2
        g2 = np.abs(self.h2_curr) ** 2
        if ordre_sic == 1:
            # Hypothèse : U2 est fort donc reçoit moins de puissance donc fait SIC
            g_sic  = g2
            g_weak = g1
            P_weak = alpha * self.P_max           # plus de puissance pour U1
            P_sic  = (1 - alpha) * self.P_max     # moins de puissance pour U2
        else:
            # Hypothèse : U1 est fort donc U1 fait le SIC
            g_sic  = g1
            g_weak = g2
            P_weak = (1 - alpha) * self.P_max
            P_sic  = alpha * self.P_max

        # SINR utilisateur faible (subit l'interférence, pas de SIC)
        SINR_weak = (P_weak * g_weak) / (P_sic * g_weak + self.bruit)

        # SINR SIC étape 1 : l'utilisateur fort décode d'abord le signal faible
        SINR_sic_step1 = (P_weak * g_sic) / (P_sic * g_sic + self.bruit)

        # SINR SIC étape 2 : après soustraction, décode son propre signal
        SINR_sic_step2 = (P_sic * g_sic) / self.bruit

        success_weak = (SINR_weak      >= self.Gamma_1)
        success_sic  = (SINR_sic_step1 >= self.Gamma_1) and (SINR_sic_step2 >= self.Gamma_2)

        return 1.0 if (success_weak and success_sic) else 0.0

    def step(self, action_index):
        """
        Prend un action_index ∈ [0, K_actions-1],
        décode en (alpha, ordre_sic), évalue, met à jour le canal.
        """
        alpha, ordre_sic = self._decode_action(action_index)
        reward = self._evaluate(alpha, ordre_sic)

        # Mise à jour Gauss-Markov
        noise1 = self._init_complex_channel(self.gain_moyen_U1)
        noise2 = self._init_complex_channel(self.gain_moyen_U2)
        self.h1_curr = self.rho * self.h1_curr + np.sqrt(1 - self.rho ** 2) * noise1
        self.h2_curr = self.rho * self.h2_curr + np.sqrt(1 - self.rho ** 2) * noise2

        # Observation : [alpha, feedback 1-bit]: no CSIT
        next_obs = [float(alpha), float(reward)]
        return next_obs, reward, False

    def get_oracle_reward(self):
        """
        Oracle : teste les 40 combinaisons (alpha, ordre_sic)
        sur les canaux courants. Appelé APRÈS step().
        """
        return max(
            self._evaluate(*self._decode_action(a))
            for a in range(self.K_actions)
        )