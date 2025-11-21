import wandb
import os
import numpy as np
import jax
import jax.random as jr
import functools as ft
import jax.numpy as jnp
import jax.tree_util as jtu

from time import time

from tqdm import tqdm
from typing import Callable

from .data import Rollout
from .utils import eval_rollout
from ..env import MultiAgentEnv
from ..algo.base import Algorithm
from ..utils.typing import Array, PRNGKey


class Trainer:

    def __init__(
            self,
            env: MultiAgentEnv,
            env_test: MultiAgentEnv,
            algo: Algorithm,
            gamma: float,
            n_env_train_per_gpu: int,
            n_env_eval_per_gpu: int,
            log_dir: str,
            seed: int,
            params: dict,
            save_log: bool = True,
            num_gpu: int = 1
    ):
        self.env = env
        self.env_test = env_test
        self.algo = algo
        self.gamma = gamma
        self.n_env_train_per_gpu = n_env_train_per_gpu
        self.n_env_eval_per_gpu = n_env_eval_per_gpu
        self.log_dir = log_dir
        self.seed = seed
        self.num_gpu = num_gpu

        if Trainer._check_params(params):
            self.params = params

        # make dir for the models
        if save_log:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            self.model_dir = os.path.join(log_dir, 'models')
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)

        wandb.login()
        wandb.init(name=params['run_name'], project='defmarl', group=env.__class__.__name__, dir=self.log_dir)

        self.save_log = save_log

        self.iters = params['training_iters']
        self.eval_interval = params['eval_interval']
        self.eval_epi = params['eval_epi']
        self.save_interval = params['save_interval']
        self.full_eval_interval = params['full_eval_interval']

        self.update_iters = params["start_iter"]
        self.key = jax.random.PRNGKey(seed)
        self.start_iter = params["start_iter"]
        self.remain_iters = params["remaining_iters"]

    @staticmethod
    def _check_params(params: dict) -> bool:
        assert 'run_name' in params, 'run_name not found in params'
        assert 'training_iters' in params, 'training_iters not found in params'
        assert 'eval_interval' in params, 'eval_interval not found in params'
        assert params['eval_interval'] > 0, 'eval_interval must be positive'
        assert 'eval_epi' in params, 'eval_epi not found in params'
        assert params['eval_epi'] >= 1, 'eval_epi must be greater than or equal to 1'
        assert 'save_interval' in params, 'save_interval not found in params'
        assert params['save_interval'] > 0, 'save_interval must be positive'
        assert 'full_eval_interval' in params, 'full_eval_interval not found in params'
        assert params['full_eval_interval'] > 0, 'full_eval_interval must be positive'
        return True

    def train(self):
        # record start time
        start_time = time()

        # preprocess the rollout function
        init_rnn_state = self.algo.init_rnn_state

        # preprocess the test function
        zmax_fn = lambda graph, value_rnn_state, params: \
            (jnp.array([[-self.env_test.reward_min]]).repeat(self.env.num_agents, axis=0), value_rnn_state)
        zmin_fn = lambda graph, value_rnn_state, params: \
            (jnp.array([[-self.env_test.reward_max]]).repeat(self.env.num_agents, axis=0), value_rnn_state)

        def eval_fn_single(params, z_fn, key):
            act_fn = ft.partial(self.algo.act, params=params)
            z_fn = ft.partial(z_fn, params=params) if z_fn is not None else None
            return eval_rollout(
                self.env_test,
                act_fn,
                init_rnn_state,
                key,
                init_rnn_state,
                z_fn
            )

        eval_opt_fn = lambda params, keys: (jax.vmap(ft.partial(eval_fn_single, params,
            self.algo.get_opt_z if hasattr(self.algo, 'get_opt_z') else None))(keys)
        )
        eval_zmax_fn = lambda params, keys: jax.vmap(ft.partial(eval_fn_single, params, zmax_fn))(keys)
        eval_zmin_fn = lambda params, keys: jax.vmap(ft.partial(eval_fn_single, params, zmin_fn))(keys)

        eval_opt_fn = jax.jit(eval_opt_fn)
        eval_zmax_fn = jax.jit(eval_zmax_fn)  # conservative
        eval_zmin_fn = jax.jit(eval_zmin_fn) # aggressive

        # start training
        pbar = tqdm(total=self.iters,initial=self.start_iter,ncols=80)

        # 用于测试的key
        eval_key = jr.PRNGKey(self.seed)
        eval_keys = jr.split(eval_key, self.num_gpu * self.n_env_eval_per_gpu * 3)
        G_eval_opt_keys = eval_keys[:self.num_gpu * self.n_env_eval_per_gpu, :].reshape(self.num_gpu, self.n_env_eval_per_gpu, -1)
        G_eval_zmax_keys = eval_keys[self.num_gpu * self.n_env_eval_per_gpu: 2 * self.num_gpu * self.n_env_eval_per_gpu, :].reshape(
            self.num_gpu, self.n_env_eval_per_gpu, -1)
        G_eval_zmin_keys = eval_keys[2 * self.num_gpu * self.n_env_eval_per_gpu: 3 * self.num_gpu * self.n_env_eval_per_gpu, :].reshape(
            self.num_gpu, self.n_env_eval_per_gpu, -1)

        # 用于训练的key
        all_iters_keys = jr.split(self.key, self.iters+1)

        @ft.partial(jax.pmap, in_axes=(None, 0), axis_name='n_gpu')
        def opt_rollout_and_eval(params: dict, eval_keys: PRNGKey):
            eval_rollouts: Rollout = eval_opt_fn(params, eval_keys)
            bT_total_reward = eval_rollouts.rewards.sum(axis=-1)
            ba_last_reward = eval_rollouts.rewards[..., -1]
            reward_min = jax.lax.pmin(bT_total_reward.min(), axis_name='n_gpu')
            reward_max = jax.lax.pmax(bT_total_reward.max(), axis_name='n_gpu')
            reward_mean = jax.lax.pmean(bT_total_reward.mean(), axis_name='n_gpu')
            reward_final = jax.lax.pmean(ba_last_reward.mean(), axis_name='n_gpu')
            bTah_cost = eval_rollouts.costs
            cost = jax.lax.pmax(bTah_cost.max(), axis_name='n_gpu')
            unsafe_frac = jax.lax.pmean((bTah_cost.max(axis=-1).max(axis=-2) >= 1e-6).mean(), axis_name='n_gpu')
            opt_z0 = jax.lax.pmean(eval_rollouts.zs[0, 0, 0, 0], axis_name='n_gpu')  # TODO：zs的维度是多少？
            return reward_min, reward_max, reward_mean, reward_final, cost, unsafe_frac, opt_z0

        @ft.partial(jax.pmap, in_axes=(None, 0), axis_name='n_gpu')
        def zmax_rollout_and_eval(params: dict, eval_keys: PRNGKey):
            eval_rollouts: Rollout = eval_zmax_fn(params, eval_keys)
            bT_total_reward = eval_rollouts.rewards.sum(axis=-1)
            ba_last_reward = eval_rollouts.rewards[..., -1]
            reward_mean = jax.lax.pmean(bT_total_reward.mean(), axis_name='n_gpu')
            reward_final = jax.lax.pmean(ba_last_reward.mean(), axis_name='n_gpu')
            bTah_cost = eval_rollouts.costs
            cost = jax.lax.pmax(bTah_cost.max(), axis_name='n_gpu')
            unsafe_frac = jax.lax.pmean((bTah_cost.max(axis=-1).max(axis=-2) >= 1e-6).mean(), axis_name='n_gpu')
            return reward_mean, reward_final, cost, unsafe_frac

        @ft.partial(jax.pmap, in_axes=(None, 0), axis_name='n_gpu')
        def zmin_rollout_and_eval(params: dict, eval_keys: PRNGKey):
            eval_rollouts: Rollout = eval_zmin_fn(params, eval_keys)
            bT_total_reward = eval_rollouts.rewards.sum(axis=-1)
            ba_last_reward = eval_rollouts.rewards[..., -1]
            reward_mean = jax.lax.pmean(bT_total_reward.mean(), axis_name='n_gpu')
            reward_final = jax.lax.pmean(ba_last_reward.mean(), axis_name='n_gpu')
            bTah_cost = eval_rollouts.costs
            cost = jax.lax.pmax(bTah_cost.max(), axis_name='n_gpu')
            unsafe_frac = jax.lax.pmean((bTah_cost.max(axis=-1).max(axis=-2) >= 1e-6).mean(), axis_name='n_gpu')
            return  reward_mean, reward_final, cost, unsafe_frac

        for iter, iter_key in enumerate(all_iters_keys,start=self.start_iter):
            if iter == 2:
                print('debug')
            # 在eval/collect/update前断开参数追踪
            current_params = jax.lax.stop_gradient(self.algo.params)
            # evaluate the algorithm
            if iter % self.eval_interval == 0:
                # eval_params = jax.device_get(self.algo.params)
                eval_info = {}
                if iter % self.full_eval_interval == 0:
                    # full test with optimal z
                    reward_min, reward_max, reward_mean, reward_final, cost, unsafe_frac, opt_z0 = \
                        opt_rollout_and_eval(current_params, G_eval_opt_keys)
                    eval_info = eval_info | {
                        "eval/reward": float(reward_mean[0]),
                        "eval/reward_final": float(reward_final[0]),
                        "eval/cost": float(cost[0]),
                        "eval/unsafe_frac": float(unsafe_frac[0]),
                        "eval/opt_z0": float(opt_z0[0]),
                    }
                    time_since_start = time() - start_time
                    eval_verbose = (f'iter: {iter:3}, time: {time_since_start:5.0f}s, reward: {float(reward_mean[0]):9.4f}, '
                                    f'min/max reward: {float(reward_min[0]):7.2f}/{float(reward_max[0]):7.2f}, cost: {float(cost[0]):8.4f}, '
                                    f'unsafe_frac: {float(unsafe_frac[0]):6.2f}')

                    tqdm.write(eval_verbose)

                # partial test with zmin and zmax
                reward_mean_zmax, reward_final_zmax, cost_zmax, unsafe_frac_zmax = \
                    zmax_rollout_and_eval(current_params, G_eval_zmax_keys)
                reward_mean_zmin, reward_final_zmin, cost_zmin, unsafe_frac_zmin = \
                    zmin_rollout_and_eval(current_params, G_eval_zmin_keys)
                eval_info = eval_info | {
                    "eval/reward_zmax": float(reward_mean_zmax[0]),
                    "eval/reward_zmin": float(reward_mean_zmin[0]),
                    "eval/reward_final_zmax": float(reward_final_zmax[0]),
                    "eval/reward_final_zmin": float(reward_final_zmin[0]),
                    "eval/cost_zmax": float(cost_zmax[0]),
                    "eval/cost_zmin": float(cost_zmin[0]),
                    "eval/unsafe_frac_zmax": float(unsafe_frac_zmax[0]),
                    "eval/unsafe_frac_zmin": float(unsafe_frac_zmin[0]),
                }
                wandb.log(eval_info, step=self.update_iters)

            # save the model
            if self.save_log and iter % self.save_interval == 0:
                self.algo.save(os.path.join(self.model_dir), iter, params_to_save=current_params)

            # collect rollouts
            G_key_x0 = jax.random.split(iter_key, self.num_gpu * self.n_env_train_per_gpu).reshape(
                self.num_gpu, self.n_env_train_per_gpu, -1)
            rollouts = self.algo.collect(current_params, G_key_x0)

            # update the algorithm
            update_info = self.algo.update(rollouts, iter)
            wandb.log(update_info, step=self.update_iters)
            self.update_iters += 1

            pbar.update(1)