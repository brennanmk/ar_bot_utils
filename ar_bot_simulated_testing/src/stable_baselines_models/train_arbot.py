import numpy as np
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
import gymnasium as gym
from ar_bot_sim.environments.tabletop.ar_bot_tabletop_gym import ARBotTabletopGym
from stable_baselines3 import PPO

def train(
    num_episodes,
    model_save_location=None,
    training_data_location=None,
    seed=43,
    obstacle=0,
) -> tuple:
    """
    train function is used to train a model

    :param num_timesteps: how many timesteps to run
    :param obstacle: whether to spawn obstacles or not
    :param model_save_location: location to save the trained model, if none the model will not be saved
    :param training_data_location: where to save training data, if none data is not saved

    :return: a tuple consisting of two lists
    """

    actions = gym.spaces.Discrete(4)

    action_mapping = {
        0: (0.0, 0.5),
        1: (0.5, 0.0),
        2: (0.0, -0.5),
        3: (-0.5, 0.0)
    }

    env_config={
        "discrete_action_mapping": action_mapping,
        "render_simulation": True,
        "number_of_obstacles": obstacle,
        "actions": actions,
        "max_timesteps_per_episode": 2000
    }

    env = ARBotTabletopGym(
        env_config
    )

    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=num_episodes)

    model = PPO(
        "MlpPolicy", env, tensorboard_log="./ppo_arbot_tensorboard/", seed=seed, learning_rate=1e-5, gamma=0.99, clip_range=0.2, device='cuda', policy_kwargs={'net_arch':[64, 128]}
    )
    model.learn(int(1e10), callback=callback_max_episodes)

    env.close()

    del env

    # check if save location is specified, if so save last model for replaying
    if model_save_location is not None:
        model.save(model_save_location)

if __name__ == '__main__':
    train(1000)