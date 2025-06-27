"""
Experimenting with EvolutionGym.
"""
import gymnasium as gym
import evogym.envs
from evogym import sample_robot
import numpy as np


def evaluate_robot(body, connections, render=False):
    env = gym.make('Walker-v0', body=body, render_mode='human' if render else None)
    
    total_reward = 0
    obs, _ = env.reset()
    
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    env.close()
    return total_reward


def main():
    print("Random robots... Beep Boop\n")    
    for i in range(5):
        body, connections = sample_robot((5, 5))
        fitness = evaluate_robot(body, connections, render=(i == 0))  # Only render first
        print(f"Robot {i+1}: Fitness = {fitness:.2f}")


if __name__ == "__main__":
    main()