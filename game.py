import gymnasium as gym
from policy.linear import LinearPolicy
import pygmo as pg
from fire import Fire
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def simulate(env, policy, gamma):
    observation, info = env.reset()
    episode_over = False
    total_reward = 0
    while not episode_over:
        # action = env.action_space.sample()  # agent policy that uses the observation and info

        action = policy.act(observation)

        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        total_reward = gamma * total_reward + reward
    #env.close()

    return total_reward


class MyProblem:
    def __init__(self, envname, gamma):
        env = gym.make(envname, continuous=True)
        policy = LinearPolicy(env)

        self.shape = shape = policy.weights.shape
        self.lower = -1 * np.ones(shape).flatten()
        self.upper = 1 * np.ones(shape).flatten()
        self.env = env
        self.gamma = gamma
        #pass

    def get_bounds(self):
        return self.lower, self.upper

    def fitness(self, x):
        policy = LinearPolicy(self.env)
        policy.weights = x.reshape(*self.shape)
        print('ok')
        return [-simulate(self.env, policy, self.gamma)]
        


def main(envname="LunarLander-v3", popsize=2, gens=10, gamma=0.99):
    prob = pg.problem(MyProblem(envname,gamma))
    
    algo = pg.algorithm(pg.pso(gen=1))

    archi = pg.archipelago(n=8, algo=algo, prob=prob, pop_size=popsize)
    #print(archi.get_migration_type());exit()
    pbar = tqdm(range(gens))
    for i in pbar:
        pop = archi.evolve()
        archi.wait()
        best, argbest = np.max(archi.get_champions_f()), np.argmax(archi.get_champions_f())
        pbar.set_postfix({"best":best})
        
        if i%3==2:
            env = gym.make(envname, continuous=True, render_mode="human")
            policy = LinearPolicy(env)
            policy.weights = archi.get_champions_x()[argbest].reshape(*policy.weights.shape)
            simulate(env, policy, gamma)
    
if __name__ == "__main__":
    Fire(main)
