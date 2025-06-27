# evogym-playground/main.py
"""
Experimenting with EvolutionGym.
"""
import gymnasium as gym
import evogym.envs # Required to override MuJo environments
from evogym import sample_robot
import argparse
import imageio
from datetime import datetime

from src.utils.bootstrap import load_config, ensure_directories_exist
from src.utils.logger import setup_logger

def parse_arguments():
    parser = argparse.ArgumentParser(description="EvolutionGym Experiment")
    parser.add_argument('--config', type=str, default='config.json', 
                        help="Path to configuration file")
    parser.add_argument('--pop-size', type=int, default=None)
    parser.add_argument('--generations', type=int, default=None)
    parser.add_argument('--env', default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    
    # Only override if explicitly provided
    if args.pop_size is not None:
        config['population_size'] = args.pop_size
    if args.generations is not None:
        config['max_generations'] = args.generations
    if args.env is not None:
        config['env'] = args.env
    if args.render:
        config['render'] = True
    elif args.no_render:
        config['render'] = False
    # Otherwise keep what's in config.json

    return config


def evaluate_robot(body, connections, render=False, logger=None):
    env = gym.make('Walker-v0', body=body, connections=connections, 
                   render_mode='human' if render else None)
    
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

def evaluate_robot_with_video(body, connections, video_path, fps=30, logger=None):
    env = gym.make('Walker-v0', body=body, connections=connections, 
                   render_mode='rgb_array')
    
    # Set metadata before any rendering
    if not hasattr(env, 'metadata'):
        env.metadata = {}
    env.metadata['render_fps'] = fps
    
    frames = []
    total_reward = 0
    obs, _ = env.reset()
    
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        frame = env.render()
        frames.append(frame)
        
        if terminated or truncated:
            break
    
    env.close()
    
    # Save video
    imageio.mimsave(video_path, frames, fps=fps, macro_block_size=1)
    if logger:
        logger.info(f"Saved video to {video_path}")
    
    return total_reward


def main():
    logger = setup_logger()
    logger.info("Starting EvolutionGym Experiment")

    config = parse_arguments()
    ensure_directories_exist()
    
    logger.info(f"Configuration: pop_size={config['population_size']}, "
                f"generations={config['max_generations']}, env={config['env']}")
    
    # Evolution loop
    for generation in range(config['max_generations']):
        logger.info(f"Generation {generation + 1}/{config['max_generations']}")
        
        best_fitness = -float('inf')
        best_robot = None
        best_robot_idx = -1
        
        # Evaluate population
        for i in range(config['population_size']):
            body, connections = sample_robot((5, 5))
            fitness = evaluate_robot(body, connections, 
                                   render=(config['render'] and i == 0 and generation == 0), 
                                   logger=logger)
            
            logger.debug(f"Robot {i+1}: Fitness = {fitness:.2f}")
            
            # Check if this is the best so far
            if fitness > best_fitness:
                best_fitness = fitness
                best_robot = (body, connections)
                best_robot_idx = i + 1
        
        logger.info(f"Generation {generation + 1} best fitness: {best_fitness:.2f} (Robot {best_robot_idx})")
        
        # Save video of generation's best
        if config['render'] and best_robot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"output/videos/{best_fitness:+06.2f}_g{generation:02d}_{timestamp}.mp4"
            evaluate_robot_with_video(best_robot[0], best_robot[1], video_path, logger=logger)
    
    # Show the final best robot
    if config['render'] and best_robot:
        logger.info("Rendering final best robot...")
        evaluate_robot(best_robot[0], best_robot[1], render=True, logger=logger)


if __name__ == "__main__":
    main()