import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(outage_exp3, outage_drqn, outage_rdpg, outage_oracle, window=100):
    smooth = lambda x: np.convolve(x, np.ones(window)/window, mode='valid')
    print('starting')
    plt.figure(figsize=(12, 6))
    plt.plot(smooth(outage_exp3),   color='blue',   linewidth=1.5, label="EXP3")
    plt.plot(smooth(outage_drqn),   color='red',    linewidth=1.5, label="DRQN")
    plt.plot(smooth(outage_rdpg),   color='purple', linewidth=1.5, label="RDPG")
    plt.plot(smooth(outage_oracle), color='black',  linewidth=1.5, 
             linestyle='--', label="Physical Limit (Oracle)")
    plt.xlabel("Episodes")
    plt.ylabel("P(Outage)")
    plt.title("Comparison EXP3 vs DRQN vs RDPG (rhô = 0.95)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

outage_exp3   = np.load('outage_exp3.npy')
outage_drqn   = np.load('outage_drqn.npy')
outage_rdpg   = np.load('outage_rdpg.npy')
outage_oracle = np.load('outage_oracle_drqn.npy')
plot_comparison(outage_exp3, outage_drqn, outage_rdpg, outage_oracle)