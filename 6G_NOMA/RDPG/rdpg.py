
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import copy

try:
    import NOMA_Gauss_Markov_RDPG as EnvModule
    from Actor import Actor
    from Critic import Critic
    from Replay_Buffer_RDPG import RecurrentReplayBuffer
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit()

EPISODES             = 8000
MAX_STEPS            = 50
BATCH_SIZE           = 64
SEQ_LEN              = 10
GAMMA                = 0.99
TAU                  = 0.005      
LEARNING_RATE_ACTOR  = 5e-5
LEARNING_RATE_CRITIC = 5e-5
HIDDEN_DIM           = 128
CAPACITY             = 20000
UPDATES_PER_EPISODE  = 5
WARMUP_EPISODES      = 200

# Bruit d'exploration Ornstein-Uhlenbeck
OU_THETA  = 0.15
OU_SIGMA_START = 0.3
OU_SIGMA_END   = 0.01
OU_SIGMA_DECAY = 6000

OBS_SIZE    = 2   # [alpha, feedback]
ACTION_SIZE = 2   # [alpha, ordre_sic]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class OUNoise:
    """Bruit corrélé temporellement pour exploration RDPG."""
    def __init__(self, size, theta=OU_THETA, sigma=0.3):
        self.size  = size
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.zeros(self.size)

    def sample(self):
        dx = -self.theta * self.state + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state

def soft_update(source, target, tau):
    for src_p, tgt_p in zip(source.parameters(), target.parameters()):
        tgt_p.data.copy_(tau * src_p.data + (1 - tau) * tgt_p.data)

actor        = Actor(OBS_SIZE, HIDDEN_DIM, ACTION_SIZE).to(device)
actor_target = copy.deepcopy(actor).to(device)
actor_target.eval()

critic        = Critic(OBS_SIZE, HIDDEN_DIM, ACTION_SIZE).to(device)
critic_target = copy.deepcopy(critic).to(device)
critic_target.eval()

optimizer_actor  = optim.Adam(actor.parameters(),  lr=LEARNING_RATE_ACTOR)
optimizer_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE_CRITIC)

replay_buffer = RecurrentReplayBuffer(CAPACITY, SEQ_LEN)
env           = EnvModule.GaussMarkov(bruit=0.1)
noise         = OUNoise(size=ACTION_SIZE)

history_outage_prob   = []
history_oracle_outage = []
history_efficiency    = []


def train_step(ou_sigma):
    if len(replay_buffer) < BATCH_SIZE:
        return

    batch       = replay_buffer.sample(BATCH_SIZE)
    obs         = batch['obs'].to(device)         # (B, Seq, 2)
    actions     = batch['actions'].to(device)     # (B, Seq, 2)
    rewards     = batch['rewards'].to(device)     # (B, Seq, 1)
    next_obs    = batch['next_obs'].to(device)    # (B, Seq, 2)
    dones       = batch['dones'].to(device)       # (B, Seq, 1)

    with torch.no_grad():
        next_actions, _ = actor_target(next_obs)
        noise_target = torch.clamp(
            torch.randn_like(next_actions) * 0.1, -0.2, 0.2
        )
        next_actions = torch.clamp(next_actions + noise_target, 0.0, 1.0)
        target_q, _  = critic_target(next_obs, next_actions)
        target_q     = rewards + (1 - dones) * GAMMA * target_q

    current_q, _ = critic(obs, actions)
    critic_loss  = F.mse_loss(current_q, target_q)

    optimizer_critic.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    optimizer_critic.step()
    pred_actions, _ = actor(obs)
    actor_loss, _   = critic(obs, pred_actions)
    actor_loss      = -actor_loss.mean()   # maximise Q

    optimizer_actor.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    optimizer_actor.step()
    soft_update(actor,  actor_target,  TAU)
    soft_update(critic, critic_target, TAU)
for episode in range(EPISODES):
    obs          = env.reset()
    hidden_actor = None
    noise.reset()

    outage_count        = 0
    oracle_outage_count = 0
    episode_agent_reward  = 0
    episode_oracle_reward = 0
    episode_data          = []
    progress = min(1.0, episode / OU_SIGMA_DECAY)
    ou_sigma = max(OU_SIGMA_END, OU_SIGMA_START - progress * (OU_SIGMA_START - OU_SIGMA_END))
    noise.sigma = ou_sigma

    for step in range(MAX_STEPS):
        obs_tensor = torch.tensor([obs], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            action_tensor, hidden_actor = actor(obs_tensor, hidden_actor)
        action_np = action_tensor.squeeze().cpu().numpy()
        if episode < WARMUP_EPISODES:
            action_np = np.random.uniform(0, 1, size=ACTION_SIZE)
        else:
            action_np = np.clip(action_np + noise.sample(), 0.0, 1.0)

        next_obs, reward_agent, done = env.step(action_np)
        reward_oracle = env.get_oracle_reward()

        if reward_agent == 0:
            outage_count += 1
        if reward_oracle == 0:
            oracle_outage_count += 1

        episode_agent_reward  += reward_agent
        episode_oracle_reward += reward_oracle
        episode_data.append((obs, action_np, reward_agent, next_obs, float(done)))
        obs = next_obs

    replay_buffer.push(episode_data)

    if episode >= WARMUP_EPISODES:
        for _ in range(UPDATES_PER_EPISODE):
            train_step(ou_sigma)
    outage_prob   = outage_count        / MAX_STEPS
    oracle_outage = oracle_outage_count / MAX_STEPS
    efficiency    = episode_agent_reward / max(episode_oracle_reward, 1e-9)

    history_outage_prob.append(outage_prob)
    history_oracle_outage.append(oracle_outage)
    history_efficiency.append(efficiency)

    if episode % 100 == 0:
        print(f"Ep {episode:4d}/{EPISODES} | "
              f"P_out agent: {outage_prob:.3f} | "
              f"P_out oracle: {oracle_outage:.3f} | "
              f"Eff: {efficiency:.2f} | "
              f"Sigma OU: {ou_sigma:.3f}")
torch.save({
    'actor':        actor.state_dict(),
    'critic':       critic.state_dict(),
    'actor_target': actor_target.state_dict(),
    'critic_target':critic_target.state_dict(),
}, 'rdpg_model.pth')

print(f"\nP_out finale (agent)  : {np.mean(history_outage_prob[-100:]):.3f}")
print(f"P_out finale (oracle) : {np.mean(history_oracle_outage[-100:]):.3f}")
print(f"Efficacité finale     : {np.mean(history_efficiency[-100:]):.3f}")
np.save('outage_rdpg.npy',   np.array(history_outage_prob))
np.save('outage_oracle_rdpg.npy', np.array(history_oracle_outage))

def plot_results(outage_agent, outage_oracle, efficiencies, window=100):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    def smooth(x):
        return np.convolve(x, np.ones(window) / window, mode='valid') if len(x) >= window else x

    ax1.plot(smooth(outage_agent),  color='red',   linewidth=1.5, label="P_out Agent (RDPG)")
    ax1.plot(smooth(outage_oracle), color='black', linewidth=1.5, linestyle='--',
             label="Physical Limit (Oracle)")
    ax1.set_title(f"Outage Probability(rolling mean {window} episodes)")
    ax1.set_ylabel("P(Outage)")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2.plot(smooth(efficiencies), color='purple', linewidth=1.5, label="Efficacité (Agent/Oracle)")
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, label="Oracle ")
    ax2.set_title(f"Efficacité relative Agent/Oracle (moyenne glissante {window} épisodes)")
    ax2.set_xlabel("Épisodes")
    ax2.set_ylabel("Ratio")
    ax2.set_ylim(0, 1.2)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


plot_results(history_outage_prob, history_oracle_outage, history_efficiency)


if __name__ == "__main__":
    pass