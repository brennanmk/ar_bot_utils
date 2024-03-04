from ray.rllib.algorithms.ppo import PPOConfig
from ar_bot_sim.environments.tabletop.ar_bot_tabletop_gym import ARBotTabletopGym
import gymnasium as gym
from ray import tune, air, train
import requests
import numpy as np
import argparse

class TrainARBot:
    def __init__(self, args) -> None:
        self.max_iterations = args.iterations
        self.stop_reward = args.stop_return
        self.evaluation_interval = args.evaluation_interval
        self.cirriculum = args.cirriculum
        self.obstacles = args.obstacles

        actions = gym.spaces.Discrete(3)
        action_mapping = {0: (0.0, 0.25), 1: (0.5, 0.0), 2: (0.0, -0.25)}

        if args.back: action_mapping[3] = (-0.5, 0.0)

        config = (
            PPOConfig()
            .environment(
                env=ARBotTabletopGym,
                env_task_fn=self.curriculum_fn,
                env_config={
                    "discrete_action_mapping": action_mapping,
                    "render_simulation": False,
                    "number_of_obstacles": self.obstacles if not self.cirriculum else 0,
                    "actions": actions,
                    "max_timesteps_per_episode": args.timesteps,
                },
            )
            .framework("torch")
            .rollouts(num_rollout_workers=args.workers)
            .resources(num_gpus=args.gpu)
            .training(gamma=args.discount, lr=args.lr, clip_param=args.clip_param, model={"fcnet_hiddens": args.network})
            .evaluation(evaluation_interval=self.evaluation_interval, evaluation_duration=args.evaluation_duration, evaluation_config={"expore":False})
        )

        for expirement in args.num_expirements:
            results = tune.Tuner(
                "PPO",
                run_config=train.RunConfig(stop=self.eval_callback),
                param_space=config.to_dict(),
            ).fit()

            results.get_dataframe().to_csv(f"{args.expirement_name}_{expirement}.csv")

    def eval_callback(self, trial_id: str, result: dict) -> bool:
        if "evaluation" in result:
            evaluation_mean_reward = result['evaluation']['sampler_results']['episode_reward_mean']
            return evaluation_mean_reward >= self.stop_reward or result["training_iteration"] >= self.max_iterations
        return result["training_iteration"] >= self.max_iterations

    def curriculum_fn(self, train_results, task_settable_env, env_ctx):
        task = task_settable_env.get_task()

        evaluation_mean_reward = train_results['evaluation']['sampler_results']['episode_reward_mean']
        if evaluation_mean_reward >= self.stop_reward and self.cirriculum and task < self.obstacles:
            task += 1

        return task

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARBot rllib testing')
    
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use during training')
    parser.add_argument('--workers', type=int, default=1, help='number of workers to use during training')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for training')
    parser.add_argument('--discount', type=float, default=0.99, help='discount factor for training')
    parser.add_argument('--clip_param', type=float, default=0.2, help='clip for training')
    parser.add_argument('--evaluation_duration', type=int, default=20, help='evaluation duration for training')
    parser.add_argument('--evaluation_interval', type=int, default=50, help='evaluation interval for training')
    parser.add_argument('--network', nargs='+', default=[32,32], help="Size of training network")
    parser.add_argument('--obstacles', type=int, default=0, help='number of obstacles to use during training')
    parser.add_argument('--timesteps', type=int, default=400, help='number of timesteps per episode to use during training')
    parser.add_argument('--stop_return', type=float, default="-4.0", help='evaluation reward to stop training')
    parser.add_argument('--iterations', type=int, default=20000, help='number of iterations total to use during training')
    parser.add_argument('--num_expirements', type=int, default=1, help='number of expirements to collect')
    parser.add_argument('--expirement_name', type=str, default=None, help='name of expirement to save csv to')

    parser.add_argument('--back', help='If true, bot can go back', default=False)
    parser.add_argument('--cirriculum', help='If true, train with cirr', default=False)

    args = parser.parse_args()

    TrainARBot(args)