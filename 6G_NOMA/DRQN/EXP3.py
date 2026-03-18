# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import NOMA_Gauss_Markov_DRQN as EnvModule


class EXP3:
    def __init__(self, K, gamma):
        self.K       = K
        self.gamma   = gamma
        self.weights = np.ones(K)

    def select_action(self):
        total = np.sum(self.weights)
        probs = (1 - self.gamma) * (self.weights / total) + self.gamma / self.K
        action = np.random.choice(self.K, p=probs)
        return action, probs

    def update(self, action, reward, probs):
        r_hat = reward / probs[action]
        self.weights[action] *= np.exp(self.gamma * r_hat / self.K)
        self.weights /= np.sum(self.weights)


# ── Fonction indépendante (hors classe) ───────────────────────────────
def run_exp3_episodes(env, n_episodes, max_steps, gamma=0.2):
    K     = env.K_actions
    agent = EXP3(K=K, gamma=gamma)

    history_outage = []
    history_oracle = []

    for episode in range(n_episodes):
        env.reset()
        action, probs = agent.select_action()   # une action par épisode

        episode_reward = 0
        episode_oracle = 0

        for step in range(max_steps):
            _, reward, _ = env.step(action)
            oracle        = env.get_oracle_reward()
            episode_reward += reward
            episode_oracle += oracle

        avg_reward = episode_reward / max_steps
        agent.update(action, avg_reward, probs)

        history_outage.append(1 - episode_reward / max_steps)
        history_oracle.append(1 - episode_oracle / max_steps)

    return np.array(history_outage), np.array(history_oracle)


# ── Simulation ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    env        = EnvModule.GaussMarkov(bruit=0.1)
    N_EPISODES = 8000
    MAX_STEPS  = 50

    outage_agent, outage_oracle = run_exp3_episodes(
        env, n_episodes=N_EPISODES, max_steps=MAX_STEPS, gamma=0.2
    )

    print(f"\nP_out agent  (EXP3)  : {np.mean(outage_agent[-100:]):.3f}")
    print(f"P_out oracle         : {np.mean(outage_oracle[-100:]):.3f}")

    np.save('outage_exp3.npy',   outage_agent)
    np.save('outage_oracle_exp3.npy', outage_oracle)
    window = 100
    smooth = lambda x: np.convolve(x, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(smooth(outage_agent),  color='blue',  linewidth=1.5, label="P_out Agent (EXP3)")
    plt.plot(smooth(outage_oracle), color='black', linewidth=1.5, linestyle='--',
             label="Limite Physique (Oracle)")
    plt.xlabel("Épisodes")
    plt.ylabel("P(Outage)")
    plt.title(f"EXP3 sur NOMA (moyenne glissante {window} épisodes)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()