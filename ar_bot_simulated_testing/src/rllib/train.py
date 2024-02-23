from ray.rllib.algorithms.ppo import PPOConfig
from ar_bot_sim.environments.tabletop.ar_bot_tabletop_gym import ARBotTabletopGym
import gymnasium as gym
from ray import tune, air, train

actions = gym.spaces.Discrete(4)

action_mapping = {0: (0.0, 0.5), 1: (0.5, 0.0), 2: (0.0, -0.5), 3: (-0.5, 0.0)}

# Create an RLlib Algorithm instance from a PPOConfig object.
config = (
    PPOConfig()
    .environment(
        env=ARBotTabletopGym,
        env_config={
            "discrete_action_mapping": action_mapping,
            "render_simulation": False,
            "number_of_obstacles": 0,
            "actions": actions,
            "max_timesteps_per_episode": 400,
        },
    )
    # Parallelize environment rollouts.
    .rollouts(num_rollout_workers=14)
    .resources(num_gpus=1)
    .training(gamma=0.99, lr=1e-5, clip_param=0.2, model={"fcnet_hiddens": [64, 64]})
)

# Train for n iterations and report results (mean episode rewards).

stopping_criteria = {"episode_reward_mean": -20}
tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop=stopping_criteria,
        checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True),
    ),
    param_space=config.to_dict(),
).fit()
