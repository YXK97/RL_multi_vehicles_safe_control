import jax.numpy as jnp
import numpy as np

from typing import Optional, Tuple

from ..utils.graph import EdgeBlock
from ..utils.typing import Action, Reward, Cost, Array, State
from .mve import MVE, MVEEnvState, MVEEnvGraphsTuple


class MVEDistMeasureTarget(MVE):
    """该任务使用直线距离作为reward和cost的度量，每个agent分配一个goal"""

    def __init__(self,
                 num_agents: int,
                 area_size: Optional[float] = None,
                 max_step: int = 1024,
                 max_travel: Optional[float] = None,
                 dt: float = 0.05,
                 params: dict = None
    ):
        super(MVEDistMeasureTarget, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)

    @property
    def reward_min(self) -> float:
        return -((jnp.linalg.norm(jnp.array([self.area_size[jnp.array([0,2])] - self.area_size[jnp.array([1,3])]])) * 0.01
                  + 0.001
                  + 2*0.02
                  + 0.00005
                  + 35**2*0.00001
                  + 30**2*0.00001) * self.max_episode_steps)

    def get_reward(self, graph: MVEEnvGraphsTuple, action: Action) -> Reward:
        num_agents = graph.env_states.agent.shape[0]
        num_goals = graph.env_states.goal.shape[0]
        assert num_agents == num_goals
        num_obsts = graph.env_states.obstacle.shape[0]

        agent_states = graph.type_states(type_idx=MVE.AGENT, n_type=num_agents)
        goals = graph.type_states(type_idx=MVE.GOAL, n_type=num_goals)
        reward = jnp.zeros(()).astype(jnp.float32)

        # goal distance reward
        agent_pos = agent_states[:, :2]
        goal_pos = goals[:, :2]
        dist2goal = jnp.linalg.norm(goal_pos - agent_pos, axis=-1)
        reward -= (dist2goal.mean()) * 0.01
        # not reaching goal reward
        reward -= jnp.where(dist2goal > self.params["dist2goal_bias"], 1.0, 0.0).mean() * 0.001

        # 航向角差异奖励
        agent_theta_grad = agent_states[:, 2] * jnp.pi/180
        agent_vec = jnp.concatenate([jnp.cos(agent_theta_grad)[:, None], jnp.sin(agent_theta_grad)[:, None]], axis=1)
        goal_theta_grad = goals[:, 2] * jnp.pi/180
        goal_vec = jnp.concatenate([jnp.cos(goal_theta_grad)[:, None], jnp.sin(goal_theta_grad)[:, None]], axis=1)
        theta2goal = jnp.einsum('ij,ij->i', agent_vec, goal_vec)
        reward += (theta2goal.mean()-1) * 0.02
        # 航向角满足要求奖励
        reward -= jnp.where(theta2goal < self.params["theta2goal_bias"], 1.0, 0.0).mean() * 0.00005

        # 速率一致性奖励
        reward -= ((action[:,0]-agent_states[:,-1]) ** 2).mean() * 0.00001

        # 前轮转角中性奖励
        reward -= (action[:,1] ** 2).mean() * 0.00001

        return reward

    def get_cost(self, graph: MVEEnvGraphsTuple) -> Cost:
        num_agents = graph.env_states.agent.shape[0]
        num_goals = graph.env_states.goal.shape[0]
        assert num_agents == num_goals
        num_obsts = graph.env_states.obstacle.shape[0]

        agent_states = graph.type_states(type_idx=MVE.AGENT, n_type=num_agents)
        obstacle_states = graph.type_states(type_idx=MVE.OBST, n_type=num_obsts)

        agent_nodes = graph.type_nodes(type_idx=MVE.AGENT, n_type=num_agents)
        agent_radius = jnp.linalg.norm(agent_nodes[0, 4:6]/2)
        if num_obsts > 0:
            obstacle_nodes = graph.type_nodes(type_idx=MVE.OBST, n_type=num_obsts)
            obst_radius = jnp.linalg.norm(obstacle_nodes[0, 4:6]/2)

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost: Array = agent_radius * 2 + self.params["collide_extra_bias"] - min_dist

        # collision between agents and obstacles
        if num_obsts == 0:
            obst_cost = jnp.zeros(num_agents)
        else:
            obstacle_pos = obstacle_states[:, :2]
            dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(obstacle_pos, 0), axis=-1)
            min_dist = jnp.min(dist, axis=1)
            obst_cost: Array = agent_radius + obst_radius + self.params["collide_extra_bias"] - min_dist

        # 对于agent是否超出边界的判断
        if "rollout_state_range" in self.params:
            if self.params["rollout_state_range"] is not None:
                rollout_state_range = self.params["rollout_state_range"]
            else:
                rollout_state_range = self.params["default_state_range"]
        else:
            rollout_state_range = self.params["default_state_range"]
        agent_bound_cost_xl = rollout_state_range[0] - agent_pos[:, 0]
        agent_bound_cost_xh = -(rollout_state_range[1] - agent_pos[:, 0])
        agent_bound_cost_yl = rollout_state_range[2] - agent_pos[:, 1]
        agent_bound_cost_yh = -(rollout_state_range[3] - agent_pos[:, 1])
        agent_bound_cost = jnp.concatenate([agent_bound_cost_xl[:, None], agent_bound_cost_xh[:, None],
                                            agent_bound_cost_yl[:, None], agent_bound_cost_yh[:, None]], axis=1)

        cost = jnp.concatenate([agent_cost[:, None], obst_cost[:, None], agent_bound_cost], axis=1)
        assert cost.shape == (num_agents, self.n_cost)

        # add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0)

        return cost

    def edge_blocks(self, state: MVEEnvState) -> list[EdgeBlock]:
        num_agents = state.agent.shape[0]
        num_goals = state.goal.shape[0]
        assert num_agents == num_goals
        num_obsts = state.obstacle.shape[0]

        # agent - agent connection
        agent_pos = state.agent[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        state_diff = state.agent[:, None, :] - state.agent[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self.params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self.params["comm_radius"])
        id_agent = jnp.arange(num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection
        agent_goal_edges = []
        for i_agent in range(num_agents):
            agent_state_i = state.agent[i_agent]
            goal_state_i = state.goal[i_agent]
            agent_goal_feats_i = agent_state_i - goal_state_i
            agent_goal_edges.append(EdgeBlock(agent_goal_feats_i[None, None, :], jnp.ones((1, 1)),
                                              jnp.array([i_agent]), jnp.array([i_agent + num_agents])))

        # agent - obstacle connection
        agent_obst_edges = []
        if num_obsts > 0:
            obs_pos = state.obstacle[:, :2]
            poss_diff = agent_pos[:, None, :] - obs_pos[None, :, :]
            dist = jnp.linalg.norm(poss_diff, axis=-1)
            agent_obs_mask = jnp.less(dist, self.params["comm_radius"])
            id_obs = jnp.arange(num_obsts) + num_agents * 2
            state_diff = state.agent[:, None, :] - state.obstacle[None, :, :]
            agent_obst_edges = [EdgeBlock(state_diff, agent_obs_mask, id_agent, id_obs)]

        return [agent_agent_edges] + agent_goal_edges + agent_obst_edges

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = self.params["rollout_state_range"][jnp.array([0,2,4,6])] + jnp.array([0,-3,0,0]) # y方向增加可行宽度（相当于增加护墙不让车跨越，让车学会不要超出道路限制）
        upper_lim = self.params["rollout_state_range"][jnp.array([1,3,5,7])] + jnp.array([0,3,0,0])
        return lower_lim, upper_lim
