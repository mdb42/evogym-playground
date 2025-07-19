# src/simulation/evaluation.py
"""
Evaluation module for EvolutionGym robots.
Keeping this separate for parallelization.
"""

import gymnasium as gym


def evaluate_phenotype(body, connections, controller=None, env_name='Walker-v0', 
                      render_mode='none', episode_steps=500, fps=30):
    # Set up environment based on render mode
    if render_mode == 'human':
        env = gym.make(env_name, body=body, connections=connections, render_mode='human')
    elif render_mode == 'video':
        env = gym.make(env_name, body=body, connections=connections, render_mode='rgb_array')
    else: # 'none'
        env = gym.make(env_name, body=body, connections=connections, render_mode=None)
    
    # Set render FPS in metadata to avoid gymnasium warning
    env.metadata['render_fps'] = fps
    
    # Run evaluation
    total_reward = 0
    obs, _ = env.reset()
    frames = [] if render_mode == 'video' else None
    
    for step in range(episode_steps):
        if controller:
            action = controller(obs)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if render_mode == 'video':
            frames.append(env.render())
        
        if terminated or truncated:
            break
    
    env.close()
    
    return total_reward, frames


def evaluate_individual(individual, **kwargs):
    """
    Wrapper for evaluating and updating an individual's fitness.
    """
    update_fitness = kwargs.pop('update_fitness', True)
    env_name = kwargs.get('env_name', 'Walker-v0')
    
    # Need a temporary env to get action space for RandomIndividual
    if hasattr(individual, 'set_action_space'):
        env = gym.make(env_name, body=individual.body, connections=individual.connections)
        individual.set_action_space(env.action_space)
        env.close()
    
    fitness, frames = evaluate_phenotype(
        individual.body,
        individual.connections,
        controller=individual.controller,
        **kwargs
    )
    
    if update_fitness:
        individual.fitness = fitness
    
    return fitness, frames
