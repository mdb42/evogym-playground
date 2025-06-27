
import imageio
import gymnasium as gym


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
