import os
import random

def _run_episodes(env, episodes, policy):

    cum_r = 0
    for _ in range(episodes):
        _, _ = env.reset()
        done = False

        while not done: 
            a = policy() #policy independant of observation
            _, r, done, trunc, _ = env.step(a)
            done = done or trunc
            cum_r +=r
    
    return cum_r/episodes #average rewards per episode

def calculate_baselines(env, log_path, max_actions, episodes=50):
    print(f"Action space : 0:{max_actions-1}")
    print(f"Calculating baselines for {episodes} episodes, expecting {episodes + episodes*(max_actions+1)} runs")
    #random action
    random_avg_r = _run_episodes(env, episodes, lambda : random.randint(0, max_actions-1))
    print(f"random action, avg_r: {random_avg_r}")
    print("Random baseline calculated")

    def policy_factor(a, epsilon):
        def policy():
            if random.random() < epsilon:
                return random.randint(0, max_actions-1)
            return a
        return policy

    #maximizing action
    max_a_avg_r = -100_000
    max_a = -1
    for a in range(max_actions):
        #small chance for a random action so env doesn't halt. In breakout we need to "shoot" sometimes 
        avg_r =  _run_episodes(env, episodes, policy_factor(a, 0.01)) 
        print(f"action: {a}, avg_r: {avg_r}")
        if avg_r > max_a_avg_r:
            max_a_avg_r = avg_r
            max_a = a

    print("Max action baseline calculated")

    #maximizing + random action    
    max_a_e_avg_r = -100_000
    max_a_e = -1
    epsilon = 0.15
    for a in range(max_actions):
        avg_r =  _run_episodes(env, episodes, policy_factor(a, epsilon))
        print(f"action: {a}, epsilon: {epsilon}, avg_r: {avg_r}")
        if avg_r > max_a_e_avg_r:
            max_a_e_avg_r = avg_r
            max_a_e = a
    
    print("Max action epsilon baseline calculated")
    os.makedirs(f'{log_path}/logs', exist_ok=True)
    with open(f'{log_path}/logs/baselines.log', 'w') as f:
        f.write(f"Random action mean r: {random_avg_r}\n")
        f.write(f"Max action {max_a} mean r: {max_a_avg_r} \n")
        f.write(f"Max action {max_a_e} epsilon {epsilon} mean r: {max_a_e_avg_r}\n")
    
    print("Baselines calculated")
