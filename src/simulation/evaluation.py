# src/simulation/evaluation.py
"""
Evaluation module for EvolutionGym robots.
"""
import gymnasium as gym
from contextlib import contextmanager
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def evaluate_individual_worker(individual, env_name, episode_steps, fps, render_mode, log_level):
    """
    Worker function with selective output suppression.
    """
    import io
    import contextlib
    
    class FilteredOutput(io.StringIO):
        def write(self, s):
            # Trying to filter out the Evolution Gym version message
            if "Evolution Gym Simulator" not in s:
                super().write(s)
                # Actually, let's watch for the unstable messages
                if "UNSTABLE" in s or "TERMINATING" in s:
                    # I know it's a physics engine issue,
                    # but I'm going to penalize the offending robots
                    pass
            return len(s)
    
    # Redirect output more safely
    filtered_out = FilteredOutput()
    filtered_err = FilteredOutput()
    
    with contextlib.redirect_stdout(filtered_out), contextlib.redirect_stderr(filtered_err):
        try:
            # Need a temporary env to get action space for RandomIndividual
            if hasattr(individual, 'set_action_space'):
                env = gym.make(env_name, body=individual.body, connections=individual.connections)
                individual.set_action_space(env.action_space)
                env.close()

            # Set up environment
            if render_mode == 'human':
                env = gym.make(env_name, body=individual.body, connections=individual.connections, render_mode='human')
            elif render_mode == 'video':
                env = gym.make(env_name, body=individual.body, connections=individual.connections, render_mode='rgb_array')
            else:
                env = gym.make(env_name, body=individual.body, connections=individual.connections, render_mode=None)

            env.metadata['render_fps'] = fps

            # Run evaluation
            total_reward = 0
            obs, _ = env.reset()
            frames = [] if render_mode == 'video' else None
            
            controller = individual.controller
            for step in range(episode_steps):
                if controller:
                    action = controller(obs)
                else:
                    action = env.action_space.sample()

                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                except Exception as e:
                    # Simulation became unstable
                    terminated = True
                    total_reward -= 100  # Take that, robot!
                
                if render_mode == 'video':
                    frames.append(env.render())
                
                if terminated or truncated:
                    break
            
            env.close()
            individual.fitness = total_reward
            
        except Exception as e:
            # Something very wrong has happened here
            individual.fitness = -1000
            
    return individual

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