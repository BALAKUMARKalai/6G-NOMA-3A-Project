
import Partitioner
import HOO
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import NOMA_Rayleigh
import NOMA_Gauss


def simple_test():

    noma_wrapper = NOMA_Rayleigh.NOMA_Adapter()
    bounds =[[0.0, 0.0], [1.0, 1.0]] 
    partitioner = Partitioner.Partitioner(min_values=bounds[0], max_values=bounds[1])
    x_armed_bandit = HOO.HOO(v1=0.4, ro=0.35, covering_generator_function=partitioner.halve_one_by_one)
     # v1 : Paramètre de régularité.
    # ro : Paramètre de découpage de l'espace.

    x_armed_bandit.set_time_horizon(max_plays=5000)   # Durée de la simulation
    x_armed_bandit.set_environment(environment_function=noma_wrapper.get_reward)
    x_armed_bandit.run_hoo()
    print("Dernière action choisie (Alpha) : {0}".format(x_armed_bandit.last_arm))
    rewards = np.array(noma_wrapper.drawn_values)
    bests = np.array(noma_wrapper.bests)
    agent_failures = 1.0 - rewards
    oracle_failures = 1.0 - bests  
    window = 500  #afficher la moyenne des 1000 derniers rounds
    outage_agent = np.convolve(agent_failures, np.ones(window)/window, mode='valid')
    outage_oracle = np.convolve(oracle_failures, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10, 5))
    # Courbe Agent
    plt.plot(outage_agent, label="Outage Probability(Agent HOO)", color='red', linewidth=1.5)
    
    plt.plot(outage_oracle, label="Physical Limit (Oracle Outage)", color='black', linestyle='--', alpha=0.6)
    
    plt.ylim(0, 1.05) 
    plt.xlabel("Rounds")
    plt.ylabel("Outage Probability(P_out)")
    plt.title("Evolution of Outage Probability (Rolling Average 500 rounds, P_max = 7W)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    
if __name__ == "__main__":
    simple_test()