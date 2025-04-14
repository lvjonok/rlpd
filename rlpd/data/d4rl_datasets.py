import gymnasium as gym

# import d4rl
import minari
import numpy as np

from rlpd.data.dataset import Dataset

"""
def qlearning_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }
"""


def minari_dataset2qlearning_dataset(minari_dataset: minari.MinariDataset):
    N = len(minari_dataset)
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    for episode in minari_dataset:
        # TODO: have to match carefully the indices
        obs = episode.observations[:-1].astype(np.float32)
        new_obs = episode.observations[1:].astype(np.float32)
        action = episode.actions.astype(np.float32)
        reward = episode.rewards.astype(np.float32)
        done_bool = episode.terminations.astype(np.bool)
        truncation_bool = episode.truncations.astype(np.bool)

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool | truncation_bool)

    return {
        "observations": np.concatenate(obs_),
        "actions": np.concatenate(action_),
        "next_observations": np.concatenate(next_obs_),
        "rewards": np.concatenate(reward_),
        "terminals": np.concatenate(done_),
    }


class D4RLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5):
        print(f"Loading dataset for {env.spec.id}...")
        original_dataset: minari.MinariDataset = minari.load_dataset(
            "mujoco/halfcheetah/expert-v0", download=True
        )

        dataset_dict = minari_dataset2qlearning_dataset(original_dataset)

        # print keys and lengths
        for k, v in dataset_dict.items():
            print(f"{k}: {v.shape}")

        # transform into qloearning dataset

        # dataset_dict = d4rl.qlearning_dataset(env)

        # print(dataset_dict["actions"])

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        dones = np.full_like(dataset_dict["rewards"], False, dtype=bool)

        for i in range(len(dones) - 1):
            if (
                np.linalg.norm(
                    dataset_dict["observations"][i + 1]
                    - dataset_dict["next_observations"][i]
                )
                > 1e-6
                or dataset_dict["terminals"][i] == 1.0
            ):
                dones[i] = True

        dones[-1] = True

        dataset_dict["masks"] = 1.0 - dataset_dict["terminals"]
        del dataset_dict["terminals"]

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict["dones"] = dones

        super().__init__(dataset_dict)


if __name__ == "__main__":
    env = gym.make("HalfCheetah-v5", max_episode_steps=100, render_mode="rgb_array")
    print(env.observation_space)
    print(env.action_space)
    dataset = D4RLDataset(env)
    print(dataset)

    trigger = lambda t: t % 10 == 0
    env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=trigger, disable_logger=True)

    # run something random till we get termination
    obs, info = env.reset()
    done = False

    import mediapy

    iteration = 0
    while not done and iteration < 105:
        action = env.action_space.sample()
        obs, reward, done, truncation, info = env.step(action)

        frame = env.render()

        print(frame.shape)

        iteration += 1
        if done or truncation:
            break
        print(f"Step {iteration} - Done: {done}")
    print("Done!")
    print(f"Episode length: {iteration}")

    env.close()
