from ray.rllib.algorithms.ppo import PPOConfig
from ar_bot_sim.environments.tabletop.ar_bot_tabletop_gym import ARBotTabletopGym
import gymnasium as gym
from ray import tune, air
actions = gym.spaces.Discrete(4)

action_mapping = {
    0: (0.0, 0.5),
    1: (0.5, 0.0),
    2: (0.0, -0.5),
    3: (-0.5, 0.0)
}

# Create an RLlib Algorithm instance from a PPOConfig object.
config = (
    PPOConfig().environment(
        env=ARBotTabletopGym,
        env_config={
            "discrete_action_mapping": action_mapping,
            "render_simulation": False,
            "number_of_obstacles": 0,
            "actions": actions,
            "max_timesteps_per_episode": 4000
        },
    )
    # Parallelize environment rollouts.
    .rollouts(num_rollout_workers=6)
)

# Construct the actual (PPO) algorithm object from the config.
algo = config.build()

# Train for n iterations and report results (mean episode rewards).

stopping_criteria = {"episode_reward_mean": 300}
tune.Tuner("PPO", run_config=air.RunConfig(stop=stopping_criteria), param_space=config.to_dict()).fit()
