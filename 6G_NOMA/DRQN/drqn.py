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
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit()

EPISODES = 5000
MAX_STEPS = 50
BATCH_SIZE = 64
SEQ_LEN = 10
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 3000
LEARNING_RATE = 5e-4
HIDDEN_DIM = 128
CAPACITY = 10000
TARGET_UPDATE_FREQ = 10
UPDATES_PER_EPISODE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

env = EnvModule.GaussMarkov(bruit=0.1) 
input_dim = 2
output_dim = env.K_actions if hasattr(env, 'K_actions') else 10 

policy_net = QNetwork(input_dim, HIDDEN_DIM, output_dim).to(device)
target_net = QNetwork(input_dim, HIDDEN_DIM, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
replay_buffer = RecurrentReplayBuffer(CAPACITY, SEQ_LEN)

history_efficiency = []
history_outage_prob = []

def train_step():
    if len(replay_buffer) < BATCH_SIZE:
        return None
    
    batch = replay_buffer.sample(BATCH_SIZE)
    
    states = batch['obs'].to(device)
    actions = batch['actions'].to(device)
    rewards = batch['rewards'].to(device)
    next_states = batch['next_obs'].to(device)
    dones = batch['dones'].to(device)
    
    q_values, _ = policy_net(states) 
    current_q = q_values.gather(2, actions)
    
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

epsilon = EPSILON_START

for episode in range(EPISODES):
    obs = env.reset() 
    hidden_state = None 
    episode_agent_reward = 0
    episode_oracle_reward = 0
    episode_data = []
    
    outage_count = 0
    total_steps = 0
    
    for step in range(MAX_STEPS):
        try:
            reward_oracle = env.get_oracle_reward()
        except AttributeError:
            reward_oracle = 1.0 
        
        if reward_oracle == 0: 
            reward_oracle = 1e-9

        obs_tensor = torch.tensor([obs], dtype=torch.float32).unsqueeze(0).to(device)
        
        if np.random.rand() < epsilon:
            action = np.random.randint(output_dim)
            with torch.no_grad():
                _, hidden_state = policy_net(obs_tensor, hidden_state)
        else:
            with torch.no_grad():
                q_values, hidden_state = policy_net(obs_tensor, hidden_state)
                action = q_values.argmax().item()
   
        next_obs, reward_agent, done = env.step(action)
        
        if reward_agent == 0:
            outage_count += 1
        total_steps += 1
        
        episode_data.append((obs, action, reward_agent, next_obs, float(done)))
        
        episode_agent_reward += reward_agent
        episode_oracle_reward += reward_oracle
        
        obs = next_obs
    
    replay_buffer.push(episode_data)
    
    for _ in range(UPDATES_PER_EPISODE):
        train_step()
    
    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())
     
    progress = min(1.0, episode / EPSILON_DECAY)
    epsilon = max(EPSILON_END, EPSILON_START - (progress * (EPSILON_START - EPSILON_END)))
    
    if episode_oracle_reward == 0: 
        episode_oracle_reward = 1e-9
    
    efficiency = episode_agent_reward / episode_oracle_reward
    history_efficiency.append(efficiency)
    
    outage_prob = outage_count / total_steps if total_steps > 0 else 0
    history_outage_prob.append(outage_prob)
    
    if episode % 100 == 0:
        print(f"Ep {episode}/{EPISODES} | Eff: {efficiency:.2f} | Outage: {outage_prob:.3f} | Eps: {epsilon:.2f}")

torch.save({
    'policy_net': policy_net.state_dict(),
    'target_net': target_net.state_dict(),
    'optimizer': optimizer.state_dict(),
}, 'drqn_model.pth')

print(f"\nEfficacité finale: {np.mean(history_efficiency[-100:]):.3f}")
print(f"Outage final: {np.mean(history_outage_prob[-100:]):.3f}")

def plot_results(efficiencies, outage_probs, window=100):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    if len(efficiencies) >= window:
        eff_smooth = np.convolve(efficiencies, np.ones(window)/window, mode='valid')
    else:
        eff_smooth = efficiencies
        
    ax1.plot(eff_smooth, label='Agent DRQN', color='purple', linewidth=1.5)
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Oracle')
    ax1.set_title(f"Efficacité (Moving Avg {window})", fontsize=14)
    ax1.set_ylabel("Ratio (Agent/Oracle)", fontsize=12)
    ax1.set_ylim(0, 1.2)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if len(outage_probs) >= window:
        outage_smooth = np.convolve(outage_probs, np.ones(window)/window, mode='valid')
    else:
        outage_smooth = outage_probs
    
    ax2.plot(outage_smooth, label='Probabilité d\'Outage', color='red', linewidth=1.5)
    ax2.set_title(f"Probabilité d'Outage (Moving Avg {window})", fontsize=14)
    ax2.set_xlabel("Episodes", fontsize=12)
    ax2.set_ylabel("P(Outage)", fontsize=12)
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_results(history_efficiency, history_outage_prob)