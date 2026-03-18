
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys

try:
    import NOMA_Gauss_Markov_DRQN as EnvModule
    from Qnetwork import QNetwork
    from Replay_Buffer import RecurrentReplayBuffer
    import EXP3
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit()

EPISODES = 8000
MAX_STEPS = 50
BATCH_SIZE = 64
SEQ_LEN = 10
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 3000
LEARNING_RATE = 5e-5
HIDDEN_DIM = 128
CAPACITY = 10000
TARGET_UPDATE_FREQ  = 50
UPDATES_PER_EPISODE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

#Environnement
env = EnvModule.GaussMarkov(bruit=0.1)
input_dim = 2                  # [alpha_t, feedback_1bit_t]
output_dim = env.K_actions      # 40 actions discrètes

#Réseaux
policy_net = QNetwork(input_dim, HIDDEN_DIM, output_dim).to(device)
target_net = QNetwork(input_dim, HIDDEN_DIM, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
replay_buffer  = RecurrentReplayBuffer(CAPACITY, SEQ_LEN)
history_outage_prob  = []
history_oracle_outage = []
history_efficiency   = []

#Training
def train_step():
    if len(replay_buffer) < BATCH_SIZE:
        return None

    batch = replay_buffer.sample(BATCH_SIZE)
    states = batch['obs'].to(device)
    actions = batch['actions'].to(device)
    rewards = batch['rewards'].to(device)
    next_states = batch['next_obs'].to(device)
    dones = batch['dones'].to(device)

    q_values, _  = policy_net(states)
    current_q    = q_values.gather(2, actions)

    with torch.no_grad():
        next_q_values, _ = target_net(next_states)
        max_next_q = next_q_values.max(dim=2, keepdim=True)[0]
        target_q = rewards + (1 - dones) * GAMMA * max_next_q

    loss = F.smooth_l1_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    return loss.item()

#Boucle principale
epsilon = EPSILON_START

for episode in range(EPISODES):
    obs          = env.reset()
    hidden_state = None

    outage_count       = 0
    oracle_outage_count = 0
    episode_agent_reward  = 0
    episode_oracle_reward = 0
    episode_data       = []

    for step in range(MAX_STEPS):
        obs_tensor = torch.tensor([obs], dtype=torch.float32).unsqueeze(0).to(device)

        # Choix de l'action (epsilon-greedy)
        if np.random.rand() < epsilon:
            action = np.random.randint(output_dim)
            with torch.no_grad():
                _, hidden_state = policy_net(obs_tensor, hidden_state)
        else:
            with torch.no_grad():
                q_values, hidden_state = policy_net(obs_tensor, hidden_state)
                action = q_values.argmax().item()

        next_obs, reward_agent, done = env.step(action)
        reward_oracle = env.get_oracle_reward()   

        if reward_agent == 0:
            outage_count += 1

        if reward_oracle == 0:
            oracle_outage_count += 1

        episode_agent_reward  += reward_agent
        episode_oracle_reward += reward_oracle
        episode_data.append((obs, action, reward_agent, next_obs, float(done)))
        obs = next_obs
        
    replay_buffer.push(episode_data)
    for _ in range(UPDATES_PER_EPISODE):
        train_step()

    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    progress = min(1.0, episode / EPSILON_DECAY)
    epsilon  = max(EPSILON_END, EPSILON_START - progress * (EPSILON_START - EPSILON_END))

    # Métriques
    outage_prob = outage_count/ MAX_STEPS
    oracle_outage = oracle_outage_count / MAX_STEPS
    efficiency = episode_agent_reward / max(episode_oracle_reward, 1e-9)

    history_outage_prob.append(outage_prob)
    history_oracle_outage.append(oracle_outage)
    history_efficiency.append(efficiency)

    if episode % 100 == 0:
        print(f"Ep {episode:4d}/{EPISODES} | "
              f"P_out agent: {outage_prob:.3f} | "
              f"P_out oracle: {oracle_outage:.3f} | "
              f"Eff: {efficiency:.2f} | "
              f"Eps: {epsilon:.2f}")


torch.save({
    'policy_net': policy_net.state_dict(),
    'target_net': target_net.state_dict(),
    'optimizer':  optimizer.state_dict(),
}, 'drqn_model.pth')

print(f"\nP_out finale (agent)  : {np.mean(history_outage_prob[-100:]):.3f}")
print(f"P_out finale (oracle) : {np.mean(history_oracle_outage[-100:]):.3f}")
print(f"Efficacité finale     : {np.mean(history_efficiency[-100:]):.3f}")
np.save('outage_drqn.npy',   np.array(history_outage_prob))
np.save('outage_oracle_drqn.npy', np.array(history_oracle_outage))
def plot_results(outage_agent, outage_oracle, efficiencies, window=100):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    def smooth(x):
        return np.convolve(x, np.ones(window) / window, mode='valid') if len(x) >= window else x

    # Plot 1 : Probabilité d'outage
    ax1.plot(smooth(outage_agent),  color='red',   linewidth=1.5, label="P_out Agent (DRQN)")
    ax1.plot(smooth(outage_oracle), color='black', linewidth=1.5, linestyle='--',
             label="Limite Physique (Oracle)")
    ax1.set_title(f"Probabilité d'Outage (moyenne glissante {window} épisodes)")
    ax1.set_ylabel("P(Outage)")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.4)
    """
    # Plot 2 : Efficacité agent / oracle
    ax2.plot(smooth(efficiencies), color='purple', linewidth=1.5, label="Efficacité (Agent/Oracle)")
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, label="Oracle (référence)")
    ax2.set_title(f"Efficacité relative Agent/Oracle (moyenne glissante {window} épisodes)")
    ax2.set_xlabel("Épisodes")
    ax2.set_ylabel("Ratio")
    ax2.set_ylim(0, 1.2)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.4)
"""
    plt.tight_layout()
    plt.show()

plot_results(history_outage_prob, history_oracle_outage, history_efficiency)