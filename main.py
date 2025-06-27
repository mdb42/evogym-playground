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
import numpy as np
import random

from src.utils.bootstrap import load_config, ensure_directories_exist
from src.utils.logger import setup_logger

def parse_arguments():
    parser = argparse.ArgumentParser(description="EvolutionGym Experiment")
    parser.add_argument('--config', type=str, default='config.json', 
                        help="Path to configuration file")
    
    parser.add_argument('--pop-size', type=int, default=None,
                        help="Population size")
    parser.add_argument('--generations', type=int, default=None,
                        help="Number of generations")
    parser.add_argument('--render', action='store_true',
                        help="Enable rendering")
    parser.add_argument('--no-render', action='store_true',
                        help="Disable rendering")
    parser.add_argument('--no-video', action='store_true',
                        help="Disable video saving even if render is on")
    parser.add_argument('--env', default=None,
                        help="Environment name (e.g., Walker-v0, Climber-v0)")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for reproducibility")
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
    if args.seed is not None:
        config['seed'] = args.seed
    if args.render:
        config['render'] = True
    elif args.no_render:
        config['render'] = False

    if args.no_video:
        config['save_videos'] = False
    
    return config


def evaluate_robot(body, connections, env_name='Walker-v0', render_mode='none', 
                  video_path=None, episode_steps=500, fps=30, logger=None):
    # Set up environment based on render mode
    if render_mode == 'human':
        env = gym.make(env_name, body=body, connections=connections, render_mode='human')
    elif render_mode == 'video':
        env = gym.make(env_name, body=body, connections=connections, render_mode='rgb_array')
        if not hasattr(env, 'metadata'):
            env.metadata = {}
        env.metadata['render_fps'] = fps
    else:  # 'none'
        env = gym.make(env_name, body=body, connections=connections, render_mode=None)
    
    # Run simulation
    total_reward = 0
    obs, _ = env.reset()
    frames = [] if render_mode == 'video' else None
    
    for step in range(episode_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if render_mode == 'video':
            frames.append(env.render())
        
        if terminated or truncated:
            break
    
    env.close()
    
    # Save video if applicable
    if render_mode == 'video' and video_path and frames:
        imageio.mimsave(video_path, frames, fps=fps, macro_block_size=1)
        if logger:
            logger.info(f"Saved video to {video_path}")
    
    return total_reward

def is_connected(body):
    """Check if all non-empty voxels in body are connected"""
    if np.sum(body > 0) == 0:
        return False
    
    # Find all non-empty positions
    positions = np.argwhere(body > 0)
    if len(positions) == 0:
        return False
    
    # Flood fill from first non-empty position
    visited = set()
    to_visit = [tuple(positions[0])]
    
    while to_visit:
        pos = to_visit.pop()
        if pos in visited:
            continue
        visited.add(pos)
        
        # Check 4-connected neighbors
        i, j = pos
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < body.shape[0] and 0 <= nj < body.shape[1]:
                if body[ni, nj] > 0 and (ni, nj) not in visited:
                    to_visit.append((ni, nj))
    
    # Check if all non-empty positions were visited
    return len(visited) == len(positions)


def mutate_robot(body, connections, mutation_rate=0.1, mutation_amount=0.3):
    """Simple mutation of robot morphology"""
    body_copy = body.copy()
    
    # Try mutation up to 10 times (might be invalid)
    for attempt in range(10):
        temp_body = body_copy.copy()
        
        # Mutate some voxels
        for i in range(temp_body.shape[0]):
            for j in range(temp_body.shape[1]):
                if random.random() < mutation_rate:
                    current = temp_body[i, j]
                    if current == 0:  # Empty - maybe add something
                        if random.random() < mutation_amount:
                            temp_body[i, j] = random.choice([1, 2, 3, 4])
                    else:  # Non-empty - maybe change or remove
                        if random.random() < mutation_amount:
                            temp_body[i, j] = 0
                        else:
                            temp_body[i, j] = random.choice([1, 2, 3, 4])
        
        # Check if result is valid
        if is_connected(temp_body):
            return temp_body, None
    
    # If not a valid mutation, return original
    return body.copy(), None

def create_next_generation(population, fitnesses, config):
    """Create next generation (tournament selection)"""
    new_population = []
    
    # Elitism - keep best N unchanged
    sorted_indices = np.argsort(fitnesses)[::-1]  # Best to worst
    for i in range(config['elitism']):
        if i < len(population):
            new_population.append(population[sorted_indices[i]])
    
    # Fill rest with mutated offspring
    while len(new_population) < config['population_size']:
        # Tournament selection
        tournament_indices = random.sample(range(len(population)), config['tournament_size'])
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        
        # Mutate winner
        parent = population[winner_idx]
        child_body, child_connections = mutate_robot(
            parent[0], parent[1], 
            config['mutation_rate'], 
            config['mutation_amount']
        )
        new_population.append((child_body, child_connections))
    
    return new_population

def main():
    logger = setup_logger()
    logger.info("Starting EvolutionGym Experiment")

    config = parse_arguments()
    ensure_directories_exist()
    
    # Set random seed if provided
    if 'seed' in config:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        logger.info(f"Random seed set to {config['seed']}")
    
    logger.info(f"Configuration: pop_size={config['population_size']}, "
                f"generations={config['max_generations']}, env={config['env']}")
    
    # Initialize population tracking
    population = []
    fitnesses = []
    
    # Evolution loop
    for generation in range(config['max_generations']):
        logger.info(f"\n=== Generation {generation + 1}/{config['max_generations']} ===")
        
        # Create population (random for first generation, evolved after)
        if generation == 0:
            population = [sample_robot(config['robot_size']) for _ in range(config['population_size'])]
        else:
            population = create_next_generation(population, fitnesses, config)
        
        # Evaluate population
        fitnesses = []
        best_fitness = -float('inf')
        best_robot = None
        best_robot_idx = -1
        
        for i, (body, connections) in enumerate(population):
            # Only render first robot of first generation
            show_robot = config['render'] and i == 0 and generation == 0
            
            fitness = evaluate_robot(
                body, connections, 
                env_name=config['env'],
                render_mode='human' if show_robot else 'none',
                episode_steps=config['episode_steps'],
                fps=config['video_fps']
            )
            
            fitnesses.append(fitness)
            logger.debug(f"Robot {i+1}: Fitness = {fitness:.2f}")
            
            # Check if this is the best so far
            if fitness > best_fitness:
                best_fitness = fitness
                best_robot = (body, connections)
                best_robot_idx = i + 1
        
        # Log generation statistics
        avg_fitness = np.mean(fitnesses)
        logger.info(f"Generation {generation + 1} - Best: {best_fitness:.2f} (Robot {best_robot_idx}), Avg: {avg_fitness:.2f}")
        
        # Save video of generation's best
        if config['save_videos'] and config['render'] and best_robot:
            timestamp = datetime.now().strftime("%H%M%S")
            video_path = f"output/videos/f{best_fitness:+07.2f}_g{generation:02d}_{timestamp}.mp4"
            evaluate_robot(
                best_robot[0], best_robot[1], 
                env_name=config['env'],
                render_mode='video',
                video_path=video_path,
                episode_steps=config['episode_steps'],
                fps=config['video_fps'],
                logger=logger
            )
        
        # Save best robot structure periodically
        if generation % config['save_best_every'] == 0 or generation == config['max_generations'] - 1:
            save_path = f"output/robots/best_g{generation:02d}_f{best_fitness:+07.2f}.npz"
            np.savez(save_path, body=best_robot[0], connections=best_robot[1])
            logger.info(f"Saved best robot to {save_path}")
    
    # Show the final best robot
    if config['render'] and best_robot:
        logger.info("\nRendering final best robot...")
        evaluate_robot(
            best_robot[0], best_robot[1], 
            env_name=config['env'],
            render_mode='human',
            episode_steps=config['episode_steps']
        )


if __name__ == "__main__":
    main()