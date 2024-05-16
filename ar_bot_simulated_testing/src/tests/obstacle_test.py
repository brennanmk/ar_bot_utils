from ar_bot_sim.environments.tabletop.ar_bot_tabletop_gym import ARBotTabletopGym
import gymnasium as gym
import pybullet as p
import time


class teleoperate:
    def __init__(self) -> None:
        """helper class to allow teleoperation of the arbot"""
        actions = gym.spaces.Discrete(5)

        action_mapping = {
            0: (0.0, 0.25),
            1: (0.5, 0.0),
            2: (0.0, -0.25),
            3: (-0.5, 0.0),
            4: (0.0, 0.0),
        }

        env_config = {
            "discrete_action_mapping": action_mapping,
            "render_simulation": True,
            "number_of_obstacles": 3,
            "actions": actions,
            "max_timesteps_per_episode": 4000,
        }

        env = ARBotTabletopGym(env_config)

        while 1:
            env.reset()
            time.sleep(1)


if __name__ == "__main__":
    teleoperate()
