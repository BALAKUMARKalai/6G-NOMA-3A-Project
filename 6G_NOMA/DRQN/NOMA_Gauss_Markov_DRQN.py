
import numpy as np
#U1 Utilisateur Faible (Loin)
#U2 Utilisateur Fort (Proche)
class GaussMarkov: 
    def __init__(self,bruit = 0.01):
        self.P_max = 1.0
        self.bruit = bruit 
        self.P_circuit = 0.1     
        
        self.gain_moyen_U1 = 0.2
        self.gain_moyen_U2 = 1.0
        self.R_target_1 = 0.5
        self.R_target_2 = 0.5
        
        self.Gamma_1 = (2**self.R_target_1) - 1
        self.Gamma_2 = (2**self.R_target_2) - 1
        self.rho = 0.95
        
        self.h1_curr = None
        self.h2_curr = None
        
        self.K_actions = 10 

    def _init_complex_channel(self, avg_gain):
        std = np.sqrt(avg_gain / 2.0)
        real = np.random.normal(0, std)
        imag = np.random.normal(0, std)
        return real + 1j * imag

    def reset(self):
        self.h1_curr = self._init_complex_channel(self.gain_moyen_U1)
        self.h2_curr = self._init_complex_channel(self.gain_moyen_U2)
        return [0.0, 0.0]

    def _calculate_reward_for_alpha(self, alpha):
        g1 = np.abs(self.h1_curr)**2 #canal h1 de gain g1
        g2 = np.abs(self.h2_curr)**2
        
        Signal_U1 = alpha * self.P_max * g1
        Interference_U1 = (1 - alpha) * self.P_max * g1 
        SINR_1 = Signal_U1 / (Interference_U1 + self.bruit)
        
        Signal_U2_decoding_U1 = alpha * self.P_max * g2 
        Interference_U2_seeing_U2 = (1 - alpha) * self.P_max * g2
        SINR_2_step1 = Signal_U2_decoding_U1 / (Interference_U2_seeing_U2 + self.bruit)
        
        signal_U2_clean = (1 - alpha) * self.P_max * g2
        SINR_2_step2 = signal_U2_clean / self.bruit
        
        success_U1 = (SINR_1 >= self.Gamma_1)
        SIC_Condition = (SINR_2_step1 >= self.Gamma_1)
        Decoding_Condition = (SINR_2_step2 >= self.Gamma_2)
        success_U2 = SIC_Condition and Decoding_Condition
        
        if success_U1 and success_U2:
            # Calcul de l'Efficacité Énergétique Globale (GEE)
            rate_1 = np.log2(1 + SINR_1)
            rate_2 = np.log2(1 + SINR_2_step2)
            sum_rate = rate_1 + rate_2
            power_total = self.P_max + self.P_circuit
            return sum_rate / power_total
        else:
            return 0.0

    def step(self, action_index): 
        alpha = (action_index + 1) / (self.K_actions + 1.0) 
        reward = self._calculate_reward_for_alpha(alpha)
        
        noise1 = self._init_complex_channel(self.gain_moyen_U1)
        noise2 = self._init_complex_channel(self.gain_moyen_U2)
        
        self.h1_curr = self.rho * self.h1_curr + np.sqrt(1 - self.rho**2) * noise1 #modélise à quel point le canal varie ou non ici rho = 0.95
        self.h2_curr = self.rho * self.h2_curr + np.sqrt(1 - self.rho**2) * noise2
    
        next_observation = [float(alpha), float(reward)]
        done = False 
        return next_observation, reward, done

    def get_oracle_reward(self):
        
        best_reward = 0.0
        
        for a_idx in range(self.K_actions):
            alpha_test = (a_idx + 1) / (self.K_actions + 1.0)
            r = self._calculate_reward_for_alpha(alpha_test)
            if r > best_reward:
                best_reward = r
                
        return best_reward

















