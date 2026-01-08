import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from typing import NamedTuple, Tuple
from defmarl.utils.typing import State, AgentState, ObstState, Action, Array, Reward, Cost, Pos2d


INF = jnp.inf

class envState(NamedTuple):
    agent: AgentState
    goal: State
    obstacle: ObstState
    dsYd_dt: Array

class lanechangeANDovertake:
    """用于给UFTSTC做测试的仿真场景，也是三车道"""

    def __init__(self, num_agents:int, num_obsts:int, dt:float, max_step:int, params:dict):

        # 环境参数
        self.num_agents = num_agents
        self.num_obsts = num_obsts
        self.dt = dt
        self.max_step = max_step

        # 其它参数
        self.params = params

        # 初始化
        self.all_goals = None
        self.all_dsYd_dts = None
        self.area_size = self.params["rollout_state_range"][:4]


    @property
    def state_dim(self) -> int:
        return 8  # x y vx vy θ dθ/dt bw bh

    @property
    def action_dim(self) -> int:
        return 2  # a：车辆纵向加速度（m/s^2） δ：前轮转角（逆时针为正，°）

    @property
    def n_cost(self) -> int:
        return 4  # agent间碰撞(1) + agent-obstacle碰撞(1) + agent超出y轴范围(高+低，2)

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions", "bound exceeds y low", "bound exceeds y high"

    @property
    def num_goals(self) -> int:
        return 3200  # 每个agent参考轨迹点的数量

    def reset(self, key: Array) -> envState:
        """使用场景类别生成函数进行agent、goal和obstacle的生成"""
        c_ycs = self.params["lane_centers"]
        xrange = self.params["default_state_range"][:2]
        yrange = self.params["default_state_range"][2:4]
        lanewidth = self.params["lane_width"]
        agents, obsts, all_goals, all_dsYd_dts = gen_scene_randomly(key, self.num_agents, self.num_goals, xrange, yrange,
                                                             lanewidth, c_ycs)
        assert obsts.shape[0] == self.num_obsts
        self.all_goals = all_goals
        self.all_dsYd_dts = all_dsYd_dts
        goals_init_indices = find_closest_goal_indices(agents, all_goals)
        agents_indices = jnp.arange(agents.shape[0])
        goals = all_goals[agents_indices, goals_init_indices, :]
        dsYd_dts = all_dsYd_dts[agents_indices, goals_init_indices, :]

        # 把bb_size加到state中
        # state: x, y, vx, vy, θ, dθ/dt, bw, bh
        agents = agents.at[:, 6:8].set(jnp.repeat(self.params["ego_bb_size"][None, :], self.num_agents, axis=0))
        goals = goals.at[:, 6:8].set(jnp.repeat(self.params["ego_bb_size"][None, :], self.num_agents, axis=0))
        obsts = obsts.at[:, 6:8].set(jnp.repeat(self.params["obst_bb_size"][None, :], self.num_obsts, axis=0))

        env_state = envState(agents, goals, obsts, dsYd_dts)

        return env_state


    def clip_state(self, state: State) -> State:
        lower_limit, upper_limit = self.state_lim
        return jnp.clip(state, lower_limit, upper_limit)


    @property
    def state_lim(self) -> Tuple[State, State]:
        """世界坐标系下的状态约束"""
        lower_lim = self.params["rollout_state_range"][jnp.array([0, 2, 4, 6, 8, 10, 12, 14])]
        upper_lim = self.params["rollout_state_range"][jnp.array([1, 3, 5, 7, 9, 11, 13, 15])]
        return lower_lim, upper_lim


    def clip_state_b(self, state: State) -> State:
        lower_limit, upper_limit = self.state_b_lim
        return jnp.clip(state, lower_limit, upper_limit)


    @property
    def state_b_lim(self) -> Tuple[State, State]:
        """车身坐标系下的状态约束"""
        lower_lim = self.params["rollout_state_b_range"][jnp.array([0, 2, 4, 6, 8, 10, 12, 14])]
        upper_lim = self.params["rollout_state_b_range"][jnp.array([1, 3, 5, 7, 9, 11, 13, 15])]
        return lower_lim, upper_lim


    def agent_step_euler(self, aS_agent_states: AgentState, ad_action: Action) -> AgentState:
        """对agent，使用3-DOF自行车动力学模型"""
        assert ad_action.shape == (self.num_agents, self.action_dim)
        assert aS_agent_states.shape == (self.num_agents, self.state_dim)
        convert_vec_s = jnp.array([1, 1, 3.6, 3.6, 180/jnp.pi, 180/jnp.pi]) # eg. km/h / convert_vec -> m/s
        convert_vec_a = jnp.array([1, 180/jnp.pi]) # m/s²不变，° / convert_vec_a -> rad

        # 参数提取
        as_S = aS_agent_states[:, :6] # x, y, vx, vy, θ, dθ/dt
        a_theta = as_S[:, 4]  # degree
        as_S_metric = as_S / convert_vec_s # km/h->m/s, degree->rad, degree/s->rad/s
        ad_action_metric = ad_action / convert_vec_a

        # 旋转矩阵计算与广义旋转矩阵构造
        a22_Q = jax.vmap(calc_2d_rot_matrix, in_axes=(0,))(a_theta)
        def construct_transform_matrix(a22_Q):
            """从 (a, 2, 2) 的旋转矩阵 Q 构造 (a, 6, 6) 的分块矩阵。"""
            a = a22_Q.shape[0]
            a66_barQ = jnp.zeros((a, 6, 6))
            a66_barQ = a66_barQ.at[:, :2, :2].set(a22_Q)
            a66_barQ = a66_barQ.at[:, 2:4, 2:4].set(a22_Q)
            two2_I = jnp.eye(2)
            a66_barQ = a66_barQ.at[:, 4:6, 4:6].set(jnp.tile(two2_I, (a, 1, 1)))
            return a66_barQ
        ass_barQ = construct_transform_matrix(a22_Q)

        # 状态量从世界坐标系向车身坐标系转换与参数提取
        as_S_b_metric = jnp.einsum('aij, ai -> aj', ass_barQ, as_S_metric)
        a_vx_b_metric = as_S_b_metric[:, 2] # m/s
        a_vy_b_metric = as_S_b_metric[:, 3] # m/s
        a_dthetadt_metric = as_S_b_metric[:, 5] # rad/s
        a_ones = jnp.ones((self.num_agents,), dtype=jnp.float32)
        m = self.params["ego_m"] # kg
        lf = self.params["ego_lf"] # m
        lr = self.params["ego_lr"] # m
        Iz = self.params["ego_Iz"] # kg*m^2
        Cf = 2*self.params["ego_Cf"] # N/rad，自行车模型需要将轮胎的侧偏刚度×2以代表轴刚度
        Cr = 2*self.params["ego_Cr"] # N/rad

        # 车辆3自由度control affine(小转向角近似)动力学模型 状态更新
        as_f = jnp.stack([a_vx_b_metric,
                          a_vy_b_metric,
                          a_vy_b_metric * a_dthetadt_metric,
                          -a_vx_b_metric * a_dthetadt_metric - (Cf+Cr)*a_vy_b_metric/(m*a_vx_b_metric) + \
                            (Cr*lr-Cf*lf)*a_dthetadt_metric/(m*a_vx_b_metric),
                          a_dthetadt_metric,
                          (Cr*lr-Cf*lf)*a_vy_b_metric/(Iz*a_vx_b_metric) - \
                            (Cf*(lf**2)+Cr*(lr**2))*a_dthetadt_metric/(Iz*a_vx_b_metric)], axis=1)
        asd_g = jnp.zeros((self.num_agents, 6, self.action_dim), dtype=jnp.float32)
        asd_g = asd_g.at[:, 2, 0].set(a_ones)
        asd_g = asd_g.at[:, 2, 1].set(Cf*(a_vy_b_metric+lr*a_dthetadt_metric)/(m*a_vx_b_metric))
        asd_g = asd_g.at[:, 3, 1].set(a_ones*Cf/m)
        asd_g = asd_g.at[:, 5, 1].set(a_ones*Cf*lf/Iz)
        as_dS_b_metric = (as_f + jnp.einsum('asd, ad -> as', asd_g, ad_action_metric)) * self.dt
        as_S_b_new_unclip_metric = as_S_b_metric + as_dS_b_metric
        as_S_b_new_unclip = as_S_b_new_unclip_metric * convert_vec_s # 公制单位转换为任务单位
        aS_S_b_new_unclip = aS_agent_states.at[:, :6].set(as_S_b_new_unclip)
        assert aS_S_b_new_unclip.shape == (self.num_agents, self.state_dim)
        aS_S_b_new = self.clip_state_b(aS_S_b_new_unclip)
        as_S_b_new = aS_S_b_new[:, :6]
        as_S_new_unclip = jnp.einsum('aij, aj -> ai', ass_barQ, as_S_b_new)
        as_S_new_unclip = as_S_new_unclip.at[:, 4].set(normalize_angle(as_S_new_unclip[:, 4]))  # θ限制在[-180, 180]°
        aS_S_new_unclip = aS_agent_states.at[:, :6].set(as_S_new_unclip)
        assert aS_S_new_unclip.shape == (self.num_agents, self.state_dim)
        aS_S_new = self.clip_state(aS_S_new_unclip)

        return aS_S_new


    def obst_step_euler(self, o_obst_states: ObstState) -> ObstState:
        """障碍车作匀速直线运动"""
        num_obsts = o_obst_states.shape[0]
        assert o_obst_states.shape == (num_obsts, self.state_dim)

        # 匀速直线运动模型
        o_x = o_obst_states[:, 0]
        o_vx = o_obst_states[:, 2]
        o_obst_states_new = o_obst_states.at[:, 0].set(o_x + o_vx/3.6*self.dt)

        assert o_obst_states_new.shape == (num_obsts, self.state_dim)
        return o_obst_states_new


    def goal_dYd_dt_step(self, aS_agent_states_new: AgentState) -> Tuple[State, Array]:
        """根据下一步的agent位置，寻找相应的距离最近的目标点作为参考，同时获取相同目标点对应的控制算法所需要的Yd一二三阶导数"""
        a_goals_indices = find_closest_goal_indices(aS_agent_states_new, self.all_goals)
        a_agents_indices = jnp.arange(aS_agent_states_new.shape[0])
        aS_goal_states = self.all_goals[a_agents_indices, a_goals_indices, :]
        a3_dsYd_dts = self.all_dsYd_dts[a_agents_indices, a_goals_indices, :]

        return aS_goal_states, a3_dsYd_dts


    def step(self, all_states: envState, action: Action) -> Tuple[envState, Reward, Cost, Pos2d]:
        # get information from graph
        agent_states = all_states.agent
        goal_states = all_states.goal
        obst_states = all_states.obstacle
        next_obst_states = self.obst_step_euler(obst_states)

        # calculate next envState
        next_agent_states = self.agent_step_euler(agent_states, action)
        next_goal_states, next_dsYd_dts = self.goal_dYd_dt_step(next_agent_states)

        # 把bb_size加到state中
        # state: x, y, vx, vy, θ, dθ/dt, bw, bh
        next_agent_states = next_agent_states.at[:, 6:8].set(jnp.repeat(self.params["ego_bb_size"][None,:], self.num_agents, axis=0))
        next_goal_states = next_goal_states.at[:, 6:8].set(jnp.repeat(self.params["ego_bb_size"][None,:], self.num_agents, axis=0))
        next_obst_states = next_obst_states.at[:, 6:8].set(jnp.repeat(self.params["obst_bb_size"][None,:], self.num_obsts, axis=0))

        goal = next_goal_states[:, :2]
        next_env_state = envState(next_agent_states, next_goal_states, next_obst_states, next_dsYd_dts)

        # calculate reward and cost
        reward = self.get_reward(all_states, action)
        cost = self.get_cost(envState)

        # calculate plot info
        # plot_info = self.get_plot_info(all_states, action) # TODO

        return next_env_state, reward, cost, goal


    def get_reward(self, all_states: envState, ad_action: Action) -> Reward:
        """和强化学习环境的reward设置保持一致"""
        aS_agents_states = all_states.agent
        aS_goals_states = all_states.goal
        # state: x, y, vx, vy, θ, dθ/dt, bw, bh
        # 参数提取
        a2_goal_pos_m = aS_goals_states[:, :2]
        a2_goal_v_kmph = aS_goals_states[:, 2:4]
        a_goal_theta_deg = aS_goals_states[:, 4]
        a2_agent_pos_m = aS_agents_states[:, :2]
        a2_agent_v_kmph = aS_agents_states[:, 2:4]
        a_agent_theta_deg = aS_agents_states[:, 4]

        # 旋转矩阵计算
        a22_Q_goal = jax.vmap(calc_2d_rot_matrix, in_axes=(0))(a_goal_theta_deg)
        a22_Q_agent = jax.vmap(calc_2d_rot_matrix, in_axes=(0))(a_agent_theta_deg)

        # 自车坐标系下的横纵向速度计算
        a2_goal_v_b_kmph = jnp.einsum('aij, ai -> aj', a22_Q_goal, a2_goal_v_kmph)
        a2_agent_v_b_kmph = jnp.einsum('aij, ai -> aj', a22_Q_agent, a2_agent_v_kmph)

        reward = jnp.zeros(()).astype(jnp.float32)
        # 循迹奖励： 位置+角度
        # 位置奖励，和目标点的欧氏距离
        a_dist = jnp.linalg.norm(a2_goal_pos_m - a2_agent_pos_m, axis=1)
        reward -= a_dist.mean() * 0.01

        # 角度奖励
        a_costheta_dist = jnp.cos((a_goal_theta_deg - a_agent_theta_deg) * jnp.pi/180)
        reward += (a_costheta_dist.mean() - 1) * 0.001

        # 速度跟踪惩罚
        a_delta_v = a2_goal_v_b_kmph[:, 0] - a2_agent_v_b_kmph[:, 0]
        reward -= jnp.abs(a_delta_v).mean() * 0.0005
        reward -= jnp.where(jnp.abs(a_delta_v) > self.params["v_bias"], 1., 0.).mean() * 0.005

        # 动作惩罚
        reward -= (ad_action[:, 0]**2).mean() * 0.00005
        reward -= (ad_action[:, 1]**2).mean() * 0.0001

        return reward

    def get_cost(self, all_states: envState) -> Cost:
        """使用射线法计算的scaling factor：α为cost的评判指标，1-α<0安全，>=0不安全，和强化学习环境的保持一致"""
        agent_states = all_states.agent
        # agent之间的scaling factor
        a_agent_cost = -jnp.ones((self.num_agents,), dtype=jnp.float32) # debug

        # agent 和 obst 之间的scaling factor
        obstacle_states = all_states.obstacle
        i_pairs, j_pairs = gen_i_j_pairs(self.num_agents, self.num_obsts)
        state_i_pairs = agent_states[i_pairs, :]
        state_j_pairs = obstacle_states[j_pairs, :]
        alpha_pairs = jax.vmap(scaling_calc, in_axes=(0, 0))(state_i_pairs, state_j_pairs)
        alpha_matrix = alpha_pairs.reshape((self.num_agents, self.num_obsts))
        a_obst_cost = jnp.max(1-alpha_matrix, axis=1)

        # agent 和 bound 之间的scaling factor，只对y方向有约束
        state_range = self.params["default_state_range"]
        yl = state_range[2]
        A = jnp.array([[0., 1.]])
        b = jnp.array([yl])
        a_bound_yl_cost = 1 - jax.vmap(scaling_calc_bound, in_axes=(0, None, None))(agent_states, A, b)

        yh = state_range[3]
        A = jnp.array([[0., -1.]])
        b = jnp.array([-yh])
        a_bound_yh_cost = 1 - jax.vmap(scaling_calc_bound, in_axes=(0, None, None))(agent_states, A, b)

        cost = jnp.stack([a_agent_cost, a_obst_cost, a_bound_yl_cost, a_bound_yh_cost], axis=1)
        assert cost.shape == (self.num_agents, self.n_cost)

        return cost


    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: Optional[dict] = None,
            n_goals: Optional[int] = None,
            **kwargs
    ) -> None:
        # ref_goals = rollout.goals
        ref_goals = rollout.env_state.goal
        n_goals = self.num_agents if n_goals is None else n_goals

        ax: Axes
        xlim = self.params["rollout_state_range"][:2]
        ylim = self.params["default_state_range"][2:4]
        fig, ax = plt.subplots(1, 1, figsize=(30,
                                (ylim[1]+3-(ylim[0]-3))*20/(xlim[1]+3-(xlim[0]-3))+4)
                               , dpi=100)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0]-3, ylim[1]+3)
        ax.set(aspect="equal")
        plt.axis("on")
        if viz_opts is None:
            viz_opts = {}

        # 画车道线
        two_yms_bold, l_yms_scatter = process_lane_marks(self.params["default_state_range"][2:4], self.params["lane_width"])
        ax.axhline(y=two_yms_bold[0], linewidth=1.5, color='b')
        ax.axhline(y=two_yms_bold[1], linewidth=1.5, color='b')
        if l_yms_scatter is not None:
            for ym in l_yms_scatter:
                ax.axhline(y=ym, linewidth=1, color='b', linestyle='--')

        # plot the first frame
        T_all_states = rollout.env_state
        all_state0 = jtu.tree_map(lambda x: x[0], T_all_states)

        agent_color = "#0068ff"
        goal_color = "#2fdd00"
        obst_color = "#8a0000"

        # state: x, y, vx, vy, θ, dθ/dt, bb_w, bb_h
        agents_state = all_state0.agent
        agents_pos = agents_state[:, :2]
        agents_theta = agents_state[:, 4]
        agents_bb_size = agents_state[:, 6:8]
        agents_radius = jnp.linalg.norm(agents_bb_size, axis=1)

        goals_state = all_state0.goal
        goals_pos = goals_state[:, :2]
        goals_theta = goals_state[:, 4]
        goals_bb_size = goals_state[:, 6:8]
        goals_radius = jnp.linalg.norm(goals_bb_size, axis=1)

        obsts_state = all_state0.obstacle
        obsts_pos = obsts_state[:, :2]
        obsts_theta = obsts_state[:, 4]
        obsts_bb_size = obsts_state[:, 6:8]
        obsts_radius = jnp.linalg.norm(obsts_bb_size, axis=1)

        # plot agents
        plot_agents_arrow = [plt.Arrow(x=agents_pos[i, 0], y=agents_pos[i, 1],
                                       dx=jnp.cos(agents_theta[i] * jnp.pi / 180) * agents_radius[i] / 2,
                                       dy=jnp.sin(agents_theta[i] * jnp.pi / 180) * agents_radius[i] / 2,
                                       width=agents_radius[i] / jnp.mean(obsts_radius),
                                       alpha=1.0, color=agent_color) for i in range(self.num_agents)]
        plot_agents_rec = [plt.Rectangle(xy=tuple(agents_pos[i, :] - agents_bb_size[i, :] / 2),
                                         width=agents_bb_size[i, 0], height=agents_bb_size[i, 1],
                                         angle=agents_theta[i], rotation_point='center',
                                         color=agent_color, linewidth=0.0, alpha=0.6) for i in range(self.num_agents)]
        col_agents = MutablePatchCollection(plot_agents_arrow + plot_agents_rec, match_original=True, zorder=6)
        ax.add_collection(col_agents)

        # plot goals
        plot_goals_arrow = [plt.Arrow(x=goals_pos[i, 0], y=goals_pos[i, 1],
                                      dx=jnp.cos(goals_theta[i] * jnp.pi / 180) * goals_radius[i] / 2,
                                      dy=jnp.sin(goals_theta[i] * jnp.pi / 180) * goals_radius[i] / 2,
                                      width=goals_radius[i] / jnp.mean(obsts_radius),
                                      alpha=0.6, color=agent_color) for i in range(self.num_agents)]
        plot_goals_rec = [plt.Rectangle(xy=tuple(goals_pos[i, :] - goals_bb_size[i, :] / 2),
                                        width=goals_bb_size[i, 0], height=goals_bb_size[i, 1],
                                        angle=goals_theta[i], rotation_point='center',
                                        color=goal_color, linewidth=0.0, alpha=0.3) for i in range(self.num_agents)]
        col_goals = MutablePatchCollection(plot_goals_arrow + plot_goals_rec, match_original=True, zorder=4)
        ax.add_collection(col_goals)

        # plot obstacles
        plot_obsts_arrow = [plt.Arrow(x=obsts_pos[i,0], y=obsts_pos[i,1],
                                        dx=jnp.cos(obsts_theta[i]*jnp.pi/180)*obsts_radius[i]/2,
                                        dy=jnp.sin(obsts_theta[i]*jnp.pi/180)*obsts_radius[i]/2,
                                        width=1, color=obst_color, alpha=1.0) for i in range(len(obsts_theta))]
        plot_obsts_rec = [plt.Rectangle(xy=tuple(obsts_pos[i,:]-obsts_bb_size[i,:]/2),
                                        width=obsts_bb_size[i,0], height=obsts_bb_size[i,1],
                                        angle=obsts_theta[i], rotation_point='center',
                                            color=obst_color, linewidth=0.0, alpha=0.6) for i in range(len(obsts_theta))]
        col_obsts = MutablePatchCollection(plot_obsts_arrow+plot_obsts_rec, match_original=True, zorder=5)
        ax.add_collection(col_obsts)



        # plot reference points
        # state: x, y, vx, vy, θ, dθ/dt, bw,
        all_ref_xs = ref_goals[:, :, 0].reshape(-1)
        all_ref_ys = ref_goals[:, :, 1].reshape(-1)
        ax.scatter(all_ref_xs, all_ref_ys, color=goal_color, zorder=7, s=5, alpha=1.0, marker='.')

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

        if "Vh" in viz_opts:
            Vh_text = ax.text(0.99, 0.99, "Vh: []", va="top", ha="right", **text_font_opts)

        # init function for animation
        def init_fn() -> list[plt.Artist]:
            return [col_agents, col_goals, col_obsts, cost_text, *safe_text, kk_text]

        def update(kk: int) -> list[plt.Artist]:
            all_state = jtu.tree_map(lambda x: x[kk], T_all_states)

            agents_state = all_state.agent
            agents_pos_t = agents_state[:, :2]
            agents_theta_t = agents_state[:, 4]
            agents_bb_size = agents_state[:, 6:8]
            agents_radius = jnp.linalg.norm(agents_bb_size, axis=1)

            goals_state = all_state.goal
            goals_pos_t = goals_state[:, :2]
            goals_theta_t = goals_state[:, 4]
            goals_bb_size = goals_state[:, 6:8]
            goals_radius = jnp.linalg.norm(goals_bb_size, axis=1)

            obsts_state = all_state.obstacle
            obsts_pos_t = obsts_state[:, :2]
            obsts_theta_t = obsts_state[:, 4]
            obsts_bb_size = obsts_state[:, 6:8]
            obsts_radius = jnp.linalg.norm(obsts_bb_size, axis=1)

            # update agents' positions
            for ii in range(self.num_agents):
                plot_agents_arrow[ii].set_data(x=agents_pos_t[ii, 0], y=agents_pos_t[ii, 1],
                                               dx=jnp.cos(agents_theta_t[ii]*jnp.pi/180)*agents_radius[ii]/2,
                                               dy=jnp.sin(agents_theta_t[ii]*jnp.pi/180)*agents_radius[ii]/2)
                plot_agents_rec[ii].set_xy(xy=tuple(agents_pos_t[ii, :]-agents_bb_size[ii, :]/2))
                plot_agents_rec[ii].set_angle(angle=agents_theta_t[ii])
            # update goals' positions
            for ii in range(self.num_agents):
                plot_goals_arrow[ii].set_data(x=goals_pos_t[ii, 0], y=goals_pos_t[ii, 1],
                                              dx=jnp.cos(goals_theta_t[ii] * jnp.pi / 180) * goals_radius[ii] / 2,
                                              dy=jnp.sin(goals_theta_t[ii] * jnp.pi / 180) * goals_radius[ii] / 2)
                plot_agents_rec[ii].set_xy(xy=tuple(goals_pos_t[ii, :] - goals_bb_size[ii, :] / 2))
                plot_agents_rec[ii].set_angle(angle=goals_theta_t[ii])
            # update obstacles' positions
            for ii in range(self.num_obsts):
                 plot_obsts_arrow[ii].set_data(x=obsts_pos_t[ii, 0], y=obsts_pos_t[ii, 1],
                                               dx=jnp.cos(obsts_theta_t[ii]*jnp.pi/180)*obsts_radius[ii]/2,
                                               dy=jnp.sin(obsts_theta_t[ii]*jnp.pi/180)*obsts_radius[ii]/2)
                 plot_obsts_rec[ii].set_xy(xy=tuple(obsts_pos_t[ii, :]-obsts_bb_size[ii, :]/2))
                 plot_obsts_rec[ii].set_angle(angle=obsts_theta_t[self.num_agents+n_goals+ii])

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

            return [col_obsts, col_agents, cost_text, *safe_text, kk_text]

        fps = 30.0
        spf = 1 / fps
        mspf = 1_000 * spf
        anim_T = len(T_all_states.agent)
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
        save_anim(ani, video_path)