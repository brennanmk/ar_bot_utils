from ray.rllib.algorithms.ppo import PPOConfig
from ar_bot_sim.environments.tabletop.ar_bot_tabletop_gym import ARBotTabletopGym
import gymnasium as gym
from ray import tune, air, train
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
    .rollouts(num_rollout_workers=12)
    .resources(num_gpus=1)
    .evaluation(
        evaluation_num_workers=1,
        evaluation_interval=20,
    )
)

# Train for n iterations and report results (mean episode rewards).

stopping_criteria = {"episode_reward_mean": 700}
tune.Tuner("PPO", run_config=air.RunConfig(stop=stopping_criteria, checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True)), param_space=config.to_dict()).fit()

save_result = aglo.save()

path_to_checkpoint = save_result.checkpoint.path
print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
)
