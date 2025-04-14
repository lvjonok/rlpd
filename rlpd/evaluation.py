from typing import Dict

import gymnasium as gym
import numpy as np

from rlpd.wrappers.wandb_video import WANDBVideo


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=num_episodes)

    for i in range(num_episodes):
        (observation, info) = env.reset()
        done, truncated = False, False
        while not done and not truncated:
            action = agent.eval_actions(observation)
            observation, _, done, truncated, _ = env.step(action)

    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}
