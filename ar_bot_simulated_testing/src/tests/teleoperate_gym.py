from ar_bot_sim.environments.tabletop.ar_bot_tabletop_gym import ARBotTabletopGym
import gymnasium as gym
import pybullet as p
import time


class teleoperate:
    def __init__(self) -> None:
        """helper class to allow teleoperation of the arbot"""
        actions = gym.spaces.Discrete(5)

        action_mapping = {
            0: (0.0, 0.5),
            1: (0.5, 0.0),
            2: (0.0, -0.5),
            3: (-0.5, 0.0),
            4: (0.0, 0.0),
        }

        env_config = {
            "discrete_action_mapping": action_mapping,
            "render_simulation": True,
            "number_of_obstacles": 0,
            "actions": actions,
            "max_timesteps_per_episode": 4000,
        }

        action = 4

        env = ARBotTabletopGym(env_config)
        env.reset()

        while 1:
            keys = p.getKeyboardEvents()

            for k, v in keys.items():
                if k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED):
                    action = 2
                if k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED):
                    action = 4
                if k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED):
                    action = 0
                if k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED):
                    action = 4

                if k == p.B3G_UP_ARROW and (v & p.KEY_WAS_TRIGGERED):
                    action = 1
                if k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED):
                    action = 4
                if k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_TRIGGERED):
                    action = 3
                if k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_RELEASED):
                    action = 4

            print(env.step(action))

            time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    teleoperate()
