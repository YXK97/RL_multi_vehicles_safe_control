import pathlib
import jax
import jax.random as jr
import jax.numpy as jnp
import functools as ft
import numpy as np

from typing import Optional, Tuple
from typing_extensions import override
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from ..trainer.data import Rollout
from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, Reward, Cost, Array, State, Path, Done, Info
from ..utils.utils import tree_index, MutablePatchCollection, save_anim
from .mve import MVE, MVEEnvState, MVEEnvGraphsTuple


class MVEPathTracking(MVE):
    """该任务使用循迹距离、方向误差和速度跟踪误差作为reward的度量，scaling factor作为cost的度量（未实现），每个agent分配一个goal并规划出一条轨迹（五次多项式）"""

    PARAMS = {
        "ego_lf": 0.905, # m
        "ego_lr": 1.305, # m
        "ego_bb_size": jnp.array([2.21, 1.48]), # bounding box的[width, height] m
        "comm_radius": 30,
        "n_obsts": 1,
        "obst_bb_size": jnp.array([4.18, 1.99]), # bounding box的[width, height] m
        "collide_extra_bias": 0.1, # 用于计算cost时避碰的margin m

        "default_state_range": jnp.array([-35., 35., -9., 9., 0., 360., 0., 30.]), # [x_l, x_u, y_l, y_u, theta_l, theta_u, v_l, v_u]
        "rollout_state_range": jnp.array([-35., 35., -9., 9., 0., 360., 0., 30.]), # rollout过程中xy坐标和theta的限制
        "agent_init_state_range": jnp.array([25., 33., -7., 7., 150., 210., 0., 0.]), # 用于agent初始化的状态范围
        "goal_state_range": jnp.array([-33., -25., -7., 7., 150., 210., 20., 20.]), # 随机生成goal时的状态范围，速度只作为目标速度
        "obst_state_range": jnp.array([-20., 20., -6., 6., 150., 210., 0., 0.]), # 随机生成obstacle的状态范围

        "dist2goal_bias": 0.1, # 用于判断agent是否到达goal m

        "theta2goal_bias": 0.98 # 用于判断agent航向角是否满足goal的要求，即agent方向向量和goal方向向量夹角的cos是否大于0.98（是否小于10度）
    }
    PARAMS.update({
        "ego_radius": jnp.linalg.norm(PARAMS["ego_bb_size"]/2), # m
        "ego_L": PARAMS["ego_lf"]+PARAMS["ego_lr"] # m
    })
    if PARAMS["n_obsts"] > 0:
        assert "obst_bb_size" in PARAMS and PARAMS["obst_bb_size"].shape == (2,)
    PARAMS.update({"obst_radius": jnp.linalg.norm(PARAMS["obst_bb_size"]/2)})

    def __init__(self,
                 num_agents: int,
                 area_size: Optional[float] = None,
                 max_step: int = 256,
                 max_travel: Optional[float] = None,
                 dt: float = 0.05,
                 params: dict = None
                 ):
        area_size = MVEPathTracking.PARAMS["default_state_range"][:4] if area_size is None else area_size
        params = MVEPathTracking.PARAMS if params is None else params
        super(MVEPathTracking, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)

    @override
    @property
    def node_dim(self) -> int:
        return 15  # state dim (4) + bb_size(2) + path_coeffs(6) + indicator(3): agent: 001, goal: 010, obstacle: 100, pad: 00-1

    @property
    def reward_min(self) -> float:
        return -(jnp.abs(self.area_size[3] - self.area_size[2])/2 * 0.01 + 1 * 0.01 + 20 * 0.01) * self.max_episode_steps * 0.7

    @override
    @property
    def n_cost(self) -> int:
        return 4 # agent间碰撞(1) + agent-obstacle碰撞(1) + agent超出y轴范围(高+低，2)

    @override
    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions", "bound exceeds y low", "bound exceeds y high"

    @override
    def reset(self, key: Array) -> GraphsTuple:
        """先生成obstacle，将obstacle视为agent，通过cost计算是否valid
        再生成agent和goal，将之前生成的obstacle还原为obstacle，利用cost计算是否valid
        最后使用五次多项式拟合初始路径"""
        state_low_idx = jnp.array([0,2,4,6])
        state_high_idx = jnp.array([1,3,5,7])

        if self.params["n_obsts"] > 0:
            # randomly generate obstacles
            def get_obst(inp):
                this_key, state_range, _ = inp
                use_key, this_key = jr.split(this_key, 2)
                return this_key, state_range, \
                        jr.uniform(use_key, (self.params["n_obsts"], self.state_dim),
                            minval=obst_state_range[state_low_idx],
                            maxval=obst_state_range[state_high_idx])

            def non_valid_obst(inp):
                "根据cost判断是否valid"
                _, _, this_candidates = inp
                empty_obsts = jnp.empty((0, self.state_dim))
                tmp_state = MVEEnvState(this_candidates, this_candidates, empty_obsts)
                paths = jnp.zeros((this_candidates.shape[0], 6))
                tmp_graph = self.get_graph(tmp_state, paths, obst_as_agent=True)
                cost = self.get_cost(tmp_graph)
                return jnp.max(cost) > -0.5

            def get_valid_obsts(state_range, key):
                use_key, this_key = jr.split(key, 2)
                # 速度均设置为0
                obst_candidates = jr.uniform(use_key, (self.params["n_obsts"], self.state_dim),
                                            minval=state_range[state_low_idx],
                                            maxval=state_range[state_high_idx])
                _, _, valid_obsts = jax.lax.while_loop(non_valid_obst, get_obst, (this_key, state_range, obst_candidates))
                return valid_obsts

            if "obst_state_range" in self.params and self.params["obst_state_range"] is not None:
                obst_state_range = self.params["obst_state_range"]
            else:
                obst_state_range = self.params["default_state_range"]
            obst_key, key = jr.split(key, 2)
            obsts = get_valid_obsts(obst_state_range, obst_key)
        else:
            obsts = jnp.empty((0, self.state_dim))

        # randomly generate agents and goals
        def get_agent_goal(inp):
            this_key, state_range, _, obsts = inp
            use_key, this_key = jr.split(this_key, 2)
            return this_key, state_range,  \
                    jr.uniform(use_key, (self.num_agents, self.state_dim),
                        minval=state_range[state_low_idx],
                        maxval=state_range[state_high_idx]), \
                    obsts

        def non_valid_agent_goal(inp):
            "根据cost判断是否valid"
            _, _, this_candidates, obsts = inp
            tmp_state = MVEEnvState(this_candidates, this_candidates, obsts)
            paths = jnp.zeros((this_candidates.shape[0], 6))
            tmp_graph = self.get_graph(tmp_state, paths)
            cost = self.get_cost(tmp_graph)
            return jnp.max(cost) > -0.5

        def get_valid_agent_goal(state_range, key, obsts):
            use_key, this_key = jr.split(key, 2)
            # 速度均设置为0
            target_candidates = jr.uniform(use_key, (self.num_agents, self.state_dim),
                                        minval=state_range[state_low_idx],
                                        maxval=state_range[state_high_idx])
            _, _, valid_targets, _ = jax.lax.while_loop(non_valid_agent_goal, get_agent_goal,
                                    (this_key, state_range, target_candidates, obsts))
            return valid_targets

        if "goal_state_range" in self.params and self.params["goal_state_range"] is not None:
            goal_state_range = self.params["goal_state_range"]
        else:
            goal_state_range = self.params["default_state_range"]
        goal_key, key = jr.split(key, 2)
        goals = get_valid_agent_goal(goal_state_range, goal_key, obsts)

        if "agent_init_state_range" in self.params:
            if self.params["agent_init_state_range"] is not None:
                agent_init_state_range = self.params["agent_init_state_range"]
            else:
                agent_init_state_range = self.params["default_state_range"]
        else:
            agent_init_state_range = self.params["default_state_range"]
        agent_key = key
        agents = get_valid_agent_goal(agent_init_state_range, agent_key, obsts)

        env_state = MVEEnvState(agents, goals, obsts)

        # 为初始agents规划五次多项式参考路径
        paths_coeff = self.generate_path(env_state)

        return self.get_graph(env_state, paths_coeff)

    @override
    def step(
            self, graph: MVEEnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[MVEEnvGraphsTuple, Reward, Cost, Done, Info]:
        # get information from graph
        agent_states = graph.type_states(type_idx=MVE.AGENT, n_type=self.num_agents)
        agent_nodes = graph.type_nodes(type_idx=MVE.AGENT, n_type=self.num_agents)
        goals = graph.type_states(type_idx=MVE.GOAL, n_type=self.num_agents)
        obstacles = graph.type_states(type_idx=MVE.OBST, n_type=self.params["n_obsts"]) if self.params["n_obsts"] > 0 else None
        paths_coeff = agent_nodes[:, 6:12]

        # calculate next graph
        action = self.transform_action(action)
        next_agent_states = self.agent_step_euler(agent_states, action)
        next_env_state = MVEEnvState(next_agent_states, goals, obstacles)
        info = {}

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # calculate reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)

        return self.get_graph(next_env_state, paths_coeff), reward, cost, done, info

    def get_reward(self, graph: MVEEnvGraphsTuple, action: Action) -> Reward:
        num_agents = graph.env_states.agent.shape[0]
        num_goals = graph.env_states.goal.shape[0]
        assert num_agents == num_goals
        num_obsts = graph.env_states.obstacle.shape[0]

        agents_states = graph.type_states(type_idx=MVE.AGENT, n_type=num_agents)
        goals_states = graph.type_states(type_idx=MVE.GOAL, n_type=num_goals)
        agents_nodes = graph.type_nodes(type_idx=MVE.AGENT, n_type=num_agents)
        reward = jnp.zeros(()).astype(jnp.float32)

        # 循迹奖励： 位置+角度
        # 位置奖励
        x = agents_states[:, 0]
        y = agents_states[:, 1]
        zeros = jnp.zeros_like(x)
        ones = jnp.ones_like(x)
        paths: Array[Path] = agents_nodes[:, 6:12]
        # jax.debug.print("path coeff={coeff}", coeff=paths)
        paths_y = (jax.vmap(lambda a, x: jnp.dot(a, x), in_axes=(0, 0))(
            paths, jnp.stack([ones, x, x**2, x**3, x**4, x**5], axis=1)))
        dist2paths = jnp.abs(paths_y - y)
        # jax.debug.print("dist2paths={dist}", dist=dist2paths)
        reward -= (dist2paths.mean()) * 0.01
        # 角度奖励
        agents_theta_grad = agents_states[:, 2] * jnp.pi / 180
        agents_vec = jnp.stack([jnp.cos(agents_theta_grad), jnp.sin(agents_theta_grad)], axis=1)
        paths_derivative = jax.vmap(lambda a, x: jnp.dot(a, x), in_axes=(0, 0))(
            paths, jnp.stack([zeros, ones, 2*x, 3*x**2, 4*x**3, 5*x**4], axis=1))
        paths_theta = jnp.atan(paths_derivative)
        paths_vec = -jnp.stack([jnp.cos(paths_theta), jnp.sin(paths_theta)], axis=1) # 这里就只能处理车辆往x轴负方向运动的情况了
        theta2paths = jnp.einsum('ij,ij->i', agents_vec, paths_vec)
        reward += (theta2paths.mean() - 1) * 0.01
        reward -= jnp.where(theta2paths < self.params["theta2goal_bias"], 1.0, 0.0).mean() * 0.005

        # 速度跟踪奖励
        v_goal = goals_states[:, 3]
        v = agents_states[:, 3]
        reward -= (jnp.abs(v_goal - v).mean()) * 0.01

        # 速率一致性奖励
        reward -= (jnp.abs(action[:, 0] - agents_states[:, -1])).mean() * 0.001

        return reward

    def get_cost(self, graph: MVEEnvGraphsTuple) -> Cost:
        num_agents = graph.env_states.agent.shape[0]
        num_goals = graph.env_states.goal.shape[0]
        assert num_agents == num_goals
        num_obsts = graph.env_states.obstacle.shape[0]

        agent_states = graph.type_states(type_idx=MVE.AGENT, n_type=num_agents)
        obstacle_states = graph.type_states(type_idx=MVE.OBST, n_type=num_obsts)

        agent_nodes = graph.type_nodes(type_idx=MVE.AGENT, n_type=num_agents)
        agent_radius = jnp.linalg.norm(agent_nodes[0, 4:6] / 2)
        if num_obsts > 0:
            obstacle_nodes = graph.type_nodes(type_idx=MVE.OBST, n_type=num_obsts)
            obst_radius = jnp.linalg.norm(obstacle_nodes[0, 4:6] / 2)

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost: Array = agent_radius * 2 + self.params["collide_extra_bias"] - min_dist

        # collision between agents and obstacles
        if num_obsts == 0:
            obst_cost = -jnp.ones(num_agents)
        else:
            obstacle_pos = obstacle_states[:, :2]
            dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(obstacle_pos, 0), axis=-1)
            min_dist = jnp.min(dist, axis=1)
            obst_cost: Array = agent_radius + obst_radius + self.params["collide_extra_bias"] - min_dist

        # 对于agent是否超出边界的判断
        if "rollout_state_range" in self.params and self.params["rollout_state_range"] is not None:
            rollout_state_range = self.params["rollout_state_range"]
        else:
            rollout_state_range = self.params["default_state_range"]
        agent_bound_cost_yl = rollout_state_range[2] - agent_pos[:, 1]
        agent_bound_cost_yh = -(rollout_state_range[3] - agent_pos[:, 1])
        agent_bound_cost = jnp.stack([agent_bound_cost_yl, agent_bound_cost_yh], axis=1)

        cost = jnp.concatenate([agent_cost[:, None], obst_cost[:, None], agent_bound_cost], axis=1)
        assert cost.shape == (num_agents, self.n_cost)

        """
        cost = jnp.concatenate([agent_cost[:, None], obst_cost[:, None]], axis=1)
        assert cost.shape == (num_agents, self.n_cost)
        """

        # add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0)

        return cost

    @override
    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: Optional[dict] = None,
            n_goals: Optional[int] = None,
            **kwargs
    ) -> None:
        n_goals = self.num_agents if n_goals is None else n_goals

        ax: Axes
        fig, ax = plt.subplots(1, 1, figsize=(20,
                                (self.area_size[3]+3-(self.area_size[2]-3))*20/(self.area_size[1]+3-(self.area_size[0]-3))+4)
                               , dpi=100)
        ax.set_xlim(self.area_size[0]-3, self.area_size[1]+3)
        ax.set_ylim(self.area_size[2]-3, self.area_size[3]+3)
        ax.set(aspect="equal")
        plt.axis("on")
        if viz_opts is None:
            viz_opts = {}

        # 画y轴方向的限制，即车道边界限制
        ax.axhline(y=self.area_size[2], linewidth=2, color='k')
        ax.axhline(y=self.area_size[3], linewidth=2, color='k')

        # 画x轴方向的限制
        ax.axvline(x=self.area_size[0], linewidth=2, color='k')
        ax.axvline(x=self.area_size[1], linewidth=2, color='k')

        # plot the first frame
        T_graph = rollout.graph
        graph0 = tree_index(T_graph, 0)

        agent_color = "#0068ff"
        goal_color = "#2fdd00"
        obst_color = "#8a0000"
        edge_goal_color = goal_color

        # plot obstacles
        if self.params["n_obsts"] > 0:
            obsts_state_bbsize = graph0.type_nodes(type_idx=MVE.OBST, n_type=self.params["n_obsts"])[:, :6]  # [n_obsts, 6] x,y,theta,v,width,height
            obsts_pos = obsts_state_bbsize[:, :2]
            obsts_theta = obsts_state_bbsize[:, 2]
            obsts_bb_size = obsts_state_bbsize[:, 4:6]
            obsts_radius = jnp.linalg.norm(obsts_bb_size, axis=1)
            plot_obsts_arrow = [plt.Arrow(x=obsts_pos[i,0], y=obsts_pos[i,1],
                                          dx=jnp.cos(obsts_theta[i]*jnp.pi/180)*obsts_radius[i]/2,
                                          dy=jnp.sin(obsts_theta[i]*jnp.pi/180)*obsts_radius[i]/2,
                                          width=1, color=obst_color, alpha=1.0) for i in range(len(obsts_theta))]
            plot_obsts_rec = [plt.Rectangle(xy=tuple(obsts_pos[i,:]-obsts_bb_size[i,:]/2),
                                            width=obsts_bb_size[i,0], height=obsts_bb_size[i,1],
                                            angle=obsts_theta[i], rotation_point='center',
                                            color=obst_color, linewidth=0.0, alpha=0.6) for i in range(len(obsts_theta))]
            plot_obsts_cir = [plt.Circle(xy=(obsts_pos[i,0], obsts_pos[i,1]), radius=self.params["obst_radius"],
                                         color=obst_color, linewidth=0.0, alpha=0.3) for i in range(len(obsts_theta))]
            col_obsts = MutablePatchCollection(plot_obsts_arrow+plot_obsts_rec+plot_obsts_cir, match_original=True, zorder=5)
            ax.add_collection(col_obsts)

        # plot agents
        agents_node = graph0.type_nodes(type_idx=MVE.AGENT, n_type=self.num_agents)
        agents_state_bbsize = agents_node[:, :6]
        agents_pos = agents_state_bbsize[:, :2]
        agents_theta = agents_state_bbsize[:, 2]
        agents_bb_size = agents_state_bbsize[:, 4:6]
        agents_radius = jnp.linalg.norm(agents_bb_size, axis=1)
        plot_agents_arrow = [plt.Arrow(x=agents_pos[i, 0], y=agents_pos[i, 1],
                                       dx=jnp.cos(agents_theta[i] * jnp.pi / 180) * agents_radius[i]/2,
                                       dy=jnp.sin(agents_theta[i] * jnp.pi / 180) * agents_radius[i]/2,
                                       width=agents_radius[i] / jnp.mean(obsts_radius),
                                       alpha=1.0, color=agent_color) for i in range(self.num_agents)]
        plot_agents_rec = [plt.Rectangle(xy=tuple(agents_pos[i,:]-agents_bb_size[i,:]/2),
                                         width=agents_bb_size[i,0], height=agents_bb_size[i,1],
                                         angle=agents_theta[i], rotation_point='center',
                                         color=agent_color, linewidth=0.0, alpha=0.6) for i in range(self.num_agents)]
        plot_agents_cir = [plt.Circle(xy=(agents_pos[i,0], agents_pos[i,1]), radius=self.params["ego_radius"],
                                      color=agent_color, linewidth=0.0, alpha=0.3) for i in range(self.num_agents)]
        col_agents = MutablePatchCollection(plot_agents_arrow+plot_agents_rec+plot_agents_cir, match_original=True, zorder=6)
        ax.add_collection(col_agents)

        # 画出agent的五次多项式path
        goals_state_bbsize = graph0.type_nodes(type_idx=MVE.GOAL, n_type=n_goals)[:, :6]
        goals_pos = goals_state_bbsize[:, :2]
        agents_path = agents_node[:, 6:12]
        a_xs = jax.vmap(lambda xl, xh: jnp.linspace(xl, xh, 100), in_axes=(0, 0))(agents_pos[:, 0], goals_pos[:, 0])
        ones = jnp.ones_like(a_xs)
        a_X = jnp.stack([ones, a_xs, a_xs**2, a_xs**3, a_xs**4, a_xs**5], axis=1)
        a_ys = jax.vmap(lambda a, x: jnp.dot(a, x), in_axes=(0, 0))(agents_path, a_X)
        path_lines = []
        for xs, ys in zip(a_xs, a_ys):
            path_lines.append(np.column_stack([xs, ys]))
        path_collection = LineCollection(path_lines, colors='k',  linewidths=1.5, linestyles='--', alpha = 1.0, zorder=7)
        ax.add_collection(path_collection)

        # plot edges
        all_pos = graph0.states[:, :2]
        edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
        is_pad = np.any(edge_index == self.num_agents + n_goals + self.params["n_obsts"], axis=0)
        e_edge_index = edge_index[:, ~is_pad]
        e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
        e_lines = np.stack([e_start, e_end], axis=1)  # (e, n_pts, dim)
        e_is_goal = (self.num_agents <= graph0.senders) & (graph0.senders < self.num_agents + n_goals)
        e_is_goal = e_is_goal[~is_pad]
        e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]
        col_edges = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
        ax.add_collection(col_edges)

        # texts
        text_font_opts = dict(
            size=16,
            color="k",
            family="sans-serif",
            weight="normal",
            transform=ax.transAxes,
        )
        cost_text = ax.text(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)
        if Ta_is_unsafe is not None:
            safe_text = [ax.text(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]
        kk_text = ax.text(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)
        z_text = ax.text(0.5, 1.04, "z: []", va="bottom", ha="center", **text_font_opts)

        # add agent labels
        label_font_opts = dict(
            size=20,
            color="k",
            family="sans-serif",
            weight="normal",
            ha="center",
            va="center",
            transform=ax.transData,
            clip_on=True,
            zorder=8,
        )
        agent_labels = [ax.text(float(agents_pos[ii, 0]), float(agents_pos[ii, 1]), f"{ii}", **label_font_opts)
                        for ii in range(self.num_agents)]

        if "Vh" in viz_opts:
            Vh_text = ax.text(0.99, 0.99, "Vh: []", va="top", ha="right", **text_font_opts)

        # init function for animation
        def init_fn() -> list[plt.Artist]:
            return [col_obsts, col_agents, col_edges, *agent_labels, cost_text, *safe_text, kk_text]

        def update(kk: int) -> list[plt.Artist]:
            graph = tree_index(T_graph, kk)
            n_pos_t = graph.states[:-1, :2] # 最后一个node是padding，不要
            n_theta_t = graph.states[:-1, 2]
            n_bb_size_t = graph.nodes[:-1, 4:6]
            n_radius = jnp.linalg.norm(n_bb_size_t, axis=1)

            # update agents' positions and labels
            for ii in range(self.num_agents):
                plot_agents_arrow[ii].set_data(x=n_pos_t[ii, 0], y=n_pos_t[ii, 1],
                                               dx=jnp.cos(n_theta_t[ii]*jnp.pi/180)*n_radius[ii]/2,
                                               dy=jnp.sin(n_theta_t[ii]*jnp.pi/180)*n_radius[ii]/2)
                plot_agents_rec[ii].set_xy(xy=tuple(n_pos_t[ii, :]-n_bb_size_t[ii, :]/2))
                plot_agents_rec[ii].set_angle(angle=n_theta_t[ii])
                plot_agents_cir[ii].set_center(xy=tuple(n_pos_t[ii, :]))
                agent_labels[ii].set_position(n_pos_t[ii, :])
            # update obstacles' positions
            if self.params["n_obsts"] > 0:
                for ii in range(self.params["n_obsts"]):
                    plot_obsts_arrow[ii].set_data(x=n_pos_t[self.num_agents+n_goals+ii, 0],
                                                  y=n_pos_t[self.num_agents+n_goals+ii, 1],
                                                  dx=jnp.cos(n_theta_t[self.num_agents+n_goals+ii]*jnp.pi/180)*n_radius[
                                                      self.num_agents+n_goals+ii]/2,
                                                  dy=jnp.sin(n_theta_t[self.num_agents+n_goals+ii]*jnp.pi/180)*n_radius[
                                                      self.num_agents+n_goals+ii]/2)
                    plot_obsts_rec[ii].set_xy(xy=tuple(n_pos_t[self.num_agents+n_goals+ii, :]-n_bb_size_t[self.num_agents+n_goals+ii, :]/2))
                    plot_obsts_rec[ii].set_angle(angle=n_theta_t[self.num_agents+n_goals+ii])
                    plot_obsts_cir[ii].set_center(xy=tuple(n_pos_t[self.num_agents+n_goals+ii, :]))

            # update edges
            e_edge_index_t = np.stack([graph.senders, graph.receivers], axis=0)
            is_pad_t = np.any(e_edge_index_t == self.num_agents + n_goals + self.params["n_obsts"], axis=0)
            e_edge_index_t = e_edge_index_t[:, ~is_pad_t]
            e_start_t, e_end_t = n_pos_t[e_edge_index_t[0, :]], n_pos_t[e_edge_index_t[1, :]]
            e_is_goal_t = (self.num_agents <= graph.senders) & (graph.senders < self.num_agents + n_goals)
            e_is_goal_t = e_is_goal_t[~is_pad_t]
            e_colors_t = [edge_goal_color if e_is_goal_t[ii] else "0.2" for ii in range(len(e_start_t))]
            e_lines_t = np.stack([e_start_t, e_end_t], axis=1)
            col_edges.set_segments(e_lines_t)
            col_edges.set_colors(e_colors_t)

            # update cost and safe labels
            if kk < len(rollout.costs):
                all_costs = ""
                for i_cost in range(rollout.costs[kk].shape[1]):
                    all_costs += f"    {self.cost_components[i_cost]}: {rollout.costs[kk][:, i_cost].max():5.4f}\n"
                all_costs = all_costs[:-2]
                cost_text.set_text(f"Cost:\n{all_costs}\nReward: {rollout.rewards[kk]:5.4f}")
            else:
                cost_text.set_text("")
            if kk < len(Ta_is_unsafe):
                a_is_unsafe = Ta_is_unsafe[kk]
                unsafe_idx = np.where(a_is_unsafe)[0]
                safe_text[0].set_text("Unsafe: {}".format(unsafe_idx))
            else:
                safe_text[0].set_text("Unsafe: {}")

            kk_text.set_text("kk={:04}".format(kk))

            # Update the z text.
            z_text.set_text(f"z: {rollout.zs[kk]}")

            if "Vh" in viz_opts:
                Vh_text.set_text(f"Vh: {viz_opts['Vh'][kk]}")

            return [col_obsts, col_agents, col_edges, *agent_labels, cost_text, *safe_text, kk_text]

        fps = 30.0
        spf = 1 / fps
        mspf = 1_000 * spf
        anim_T = len(T_graph.n_node)
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
        save_anim(ani, video_path)

    def edge_blocks(self, state: MVEEnvState) -> list[EdgeBlock]:
        num_agents = state.agent.shape[0]
        num_goals = state.goal.shape[0]
        assert num_agents == num_goals
        num_obsts = state.obstacle.shape[0]

        agent_pos = state.agent[:, :2]
        id_agent = jnp.arange(num_agents)

        # agent - agent connection
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        state_diff = state.agent[:, None, :] - state.agent[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self.params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self.params["comm_radius"])
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

    def generate_path(self, env_state: MVEEnvState) -> Path:
        """根据起点和终点求解五次多项式并写入graph"""
        agent_states = env_state.agent
        goal_states = env_state.goal
        @ft.partial(jax.jit)
        def A_b_create_and_solve(agent_state, goal_state) -> Path:
            x0 = agent_state[0]
            x1 = goal_state[0]
            A = jnp.array([[1, x0, x0**2,   x0**3,    x0**4,    x0**5],
                           [0,  1,  2*x0, 3*x0**2,  4*x0**3,  5*x0**4],
                           [0,  0,     2,    6*x0, 12*x0**2, 20*x0**3],
                           [1, x1, x1**2,   x1**3,    x1**4,    x1**5],
                           [0,  1,  2*x1, 3*x1**2,  4*x1**3,  5*x1**4],
                           [0,  0,     2,    6*x1, 12*x1**2, 20*x1**3],])
            y0 = agent_state[1]
            y1 = goal_state[1]
            t0 = agent_state[2]*jnp.pi/180
            t1 = goal_state[2]*jnp.pi/180
            b = jnp.array([y0, jnp.tan(t0), 0, y1, jnp.tan(t1), 0])
            coeff = jnp.linalg.solve(A, b)
            return coeff
        coeffs = jax.vmap(A_b_create_and_solve, in_axes=(0, 0))(agent_states, goal_states)
        return coeffs

    @override
    def get_graph(self, env_state: MVEEnvState, paths: Path, obst_as_agent:bool = False) -> MVEEnvGraphsTuple:
        num_agents = env_state.agent.shape[0]
        num_goals = env_state.goal.shape[0]
        num_obsts = env_state.obstacle.shape[0] # TODO: 为0时报错，但理论上可以为0
        assert num_agents > 0 and num_goals > 0, "至少需要设定agent和goal!"
        assert num_agents == num_goals, "每一个agent对应一个goal"
        # node features
        # states
        node_feats = jnp.zeros((num_agents + num_goals + num_obsts, self.node_dim))
        node_feats = node_feats.at[:num_agents, :self.state_dim].set(env_state.agent)
        node_feats = node_feats.at[num_agents: num_agents + num_goals, :self.state_dim].set(env_state.goal)
        if num_obsts > 0:
            node_feats = node_feats.at[num_agents + num_goals:, :self.state_dim].set(env_state.obstacle)

        # bounding box长宽
        if obst_as_agent:
            node_feats = node_feats.at[:num_agents + num_goals, 4:6].set(self.params["obst_bb_size"])
        else:
            node_feats = node_feats.at[:num_agents + num_goals, 4:6].set(self.params["ego_bb_size"])
        if num_obsts > 0:
            node_feats = node_feats.at[num_agents + num_goals:, 4:6].set(self.params["obst_bb_size"])

        # 对agent设置五次多项式路径规划
        node_feats = node_feats.at[:num_agents, 6:12].set(paths)

        # indicators
        node_feats = node_feats.at[:num_agents, 14].set(1.0)
        node_feats = node_feats.at[num_agents: num_agents + num_goals, 13].set(1.0)
        if num_obsts > 0:
            node_feats = node_feats.at[num_agents + num_goals:, 12].set(1.0)

        # node type
        node_type = -jnp.ones((num_agents + num_goals + num_obsts), dtype=jnp.int32)
        node_type = node_type.at[:num_agents].set(MVE.AGENT)
        node_type = node_type.at[num_agents: num_agents + num_goals].set(MVE.GOAL)
        if num_obsts > 0:
            node_type = node_type.at[num_agents + num_goals:].set(MVE.OBST)

        # edges
        edge_blocks = self.edge_blocks(env_state)

        # create graph
        states = jnp.concatenate([env_state.agent, env_state.goal], axis=0)
        if num_obsts > 0:
            states = jnp.concatenate([states, env_state.obstacle], axis=0)
        return GetGraph(node_feats, node_type, edge_blocks, env_state, states).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = self.params["rollout_state_range"][
            jnp.array([0, 2, 4, 6])]  + jnp.array([0,-3,0,0]) # y方向增加可行宽度（相当于增加护墙不让车跨越，让车学会不要超出道路限制）
        upper_lim = self.params["rollout_state_range"][jnp.array([1, 3, 5, 7])]  + jnp.array([0,3,0,0])
        return lower_lim, upper_lim

    @override
    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.array([0., -15.])[None, :].repeat(self.num_agents, axis=0) # v(不能倒车), delta
        upper_lim = jnp.array([30., 15.])[None, :].repeat(self.num_agents, axis=0)
        return lower_lim, upper_lim
