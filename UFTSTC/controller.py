import jax.numpy as jnp

from typing import Tuple
from defmarl.utils.typing import State, AgentState, ObstState, Action
from defmarl.utils.utils import nthroot

class UFTSTCController:
    """使用LvChen论文里的UFTSTC算法计算前轮转角控制量，去掉了转角上限估计器，改为直接对转角求饱和函数
    state: X(m) Y(m) vx(km/h) vy(km/h) Psi(deg) dPsi(gamma)(deg/s) bw(m) bh(m)
    action: ax(=0)(m/s^2) deltaf(deg/s)"""
    def __init__(self, num_agents:int, num_obsts:int, dt:float, Af: float, r:float, mu:float, c:float, k1:float,
                 k2:float, k3:float, k4:float, v:float, Delta1:float, Delta2:float, p_num:int, p_den:int, alpha:float,
                 Cf:float, Cr:float, Lf:float, Lr:float, m:float, Iz:float):

        # 缓存
        self.delta_xoe = 100 * jnp.ones((num_agents, num_obsts), dtype=jnp.float32)
        self.delta_dxoe = jnp.zeros((num_agents, num_obsts), dtype=jnp.float32)
        self.BD_int = jnp.zeros((num_agents, num_obsts), dtype=jnp.float32)

        # 算法参数设置
        self.dt = dt
        self.Af = Af
        self.r = r
        self.mu = mu
        self.c = c
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.v = v
        self.Delta1 = Delta1
        self.Delta2 = Delta2
        self.p_num = p_num
        self.p_den = p_den
        self.alpha = alpha

        # 动力学参数设置
        self.Cf = Cf
        self.Cr = Cr
        self.Lf = Lf
        self.Lr = Lr
        self.m = m
        self.Iz = Iz


    def calc_delta_xoe_dxoe_ddxoe(self, aS_agent_states: AgentState, oS_obst_states: ObstState
                                  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        a2_agent_pos = aS_agent_states[:, :2]
        o2_obst_pos = oS_obst_states[:, :2]
        # distance between agents and obstacles
        ao_delta_xoe = jnp.linalg.norm(jnp.expand_dims(a2_agent_pos, 1) - jnp.expand_dims(o2_obst_pos, 0), axis=-1)
        ao_delta_dxoe = (ao_delta_xoe - self.delta_xoe) / self.dt
        self.delta_xoe = ao_delta_xoe
        ao_delta_ddxoe = (ao_delta_dxoe - self.delta_dxoe) / self.dt
        self.delta_dxoe = ao_delta_dxoe
        return ao_delta_xoe, ao_delta_dxoe, ao_delta_ddxoe


    def calc_Uod_dUod_ddUod(self, aS_agent_states: AgentState, oS_obst_states: ObstState
                            ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        delta_xoe, delta_dxoe, delta_ddxoe = self.calc_delta_xoe_dxoe_ddxoe(aS_agent_states, oS_obst_states)
        ao_Uod = jnp.where(delta_xoe > self.mu, 0,
                    jnp.where(delta_xoe > self.r,
                        self.Af*jnp.cos(jnp.pi/2 * (delta_xoe - self.r)**(2*self.c)/((self.mu - self.r)**(2*self.c))),
                        self.Af))
        ao_dUod = jnp.where(delta_xoe > self.mu, 0,
                    jnp.where(delta_xoe > self.r,
                        -self.Af*jnp.sin(jnp.pi/2 * (delta_xoe - self.r)**(2*self.c)/((self.mu - self.r)**(2*self.c))) * \
                        jnp.pi*self.c*delta_dxoe * (delta_xoe - self.r)**(2*self.c-1) / (self.mu-self.r)**(2*self.c),
                        0))
        ao_ddUod = jnp.where(delta_xoe > self.mu, 0,
                    jnp.where(delta_xoe > self.r,
                        -self.Af*jnp.cos(jnp.pi/2 * (delta_xoe - self.r)**(2*self.c)/((self.mu - self.r)**(2*self.c))) * \
                        jnp.pi**2 * self.c**2 * delta_dxoe**2 * (delta_xoe - self.r)**(4*self.c-2) / (self.mu-self.r)**(4*self.c) \
                        -self.Af*jnp.sin(jnp.pi/2 * (delta_xoe - self.r)**(2*self.c)/((self.mu - self.r)**(2*self.c))) * \
                        jnp.pi * self.c * ((self.c*2-1) * delta_dxoe**2 * (delta_xoe-self.r)**(2*self.c-2) \
                        +delta_ddxoe * (delta_xoe-self.r)**(2*self.c-1)) / (self.mu-self.r)**(2*self.c),
                        0))
        return ao_Uod, ao_dUod, ao_ddUod


    def calc_BD_dBD_ddBD(self, aS_agent_states: AgentState, oS_obst_states: ObstState
                         ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        Uod, dUod, ddUod = self.calc_Uod_dUod_ddUod(aS_agent_states, oS_obst_states)
        ao_BD = jnp.log(self.Af**2 / (self.Af**2 - Uod**2))
        ao_dBD = 2 * Uod * dUod / (self.Af**2 - Uod**2)
        ao_ddBD = 2 * (dUod**2 + Uod*ddUod) / (self.Af**2 - Uod**2) + 4 * Uod**2 * dUod**2 / (self.Af**2 - Uod**2)**2
        return ao_BD, ao_dBD, ao_ddBD


    def integrate_BD_update(self, ao_BD:jnp.ndarray) -> jnp.ndarray:
        ao_BD_int = self.BD_int + ao_BD*self.dt
        self.BD_int = ao_BD_int
        return ao_BD_int


    def calc_Y_dY_ddY(self, aS_agent_states: AgentState, ad_action: Action
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        a_Y = aS_agent_states[:, 1]
        a_vx = aS_agent_states[:, 2] / 3.6 # metric
        a_vy = aS_agent_states[:, 3] / 3.6 # metric
        a_Psi = aS_agent_states[:, 4] * jnp.pi/180 # metric
        a_dPsi = a_gamma = aS_agent_states[:, 5] * jnp.pi/180 # metric
        a_deltaf = ad_action[:, 1] * jnp.pi/180 # metric

        a_dvy = (self.Cf+self.Cr)*a_vy / (self.m*a_vx) - self.Cf * a_deltaf / self.m \
            + ((self.Lf*self.Cf - self.Lr*self.Cr)/(self.m*a_vx) - a_vx)*a_gamma
        a_dY = a_vx * a_Psi / a_vy
        a_ddy = a_vx * a_gamma + a_dvy

        return a_Y, a_dY, a_ddy


    def calc_beta_dbeta_ddbeta_metric(self, aS_agent_states: AgentState, ad_action: Action
                               ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        a_vx = aS_agent_states[:, 2] / 3.6 # metric
        a_vy = aS_agent_states[:, 3] / 3.6 # metric
        a_dPsi = a_gamma = aS_agent_states[:, 5] * jnp.pi/180 # metric
        a_deltaf = ad_action[:, 1] * jnp.pi/180 # metric

        a_beta_metric = a_vy / a_vx
        a_dvy = (self.Cf + self.Cr) * a_vy / (self.m * a_vx) - self.Cf * a_deltaf / self.m \
                + ((self.Lf * self.Cf - self.Lr * self.Cr) / (self.m * a_vx) - a_vx) * a_gamma
        a_dbeta_metric = a_dvy / a_vx
        a_ddbeta_metric = jnp.zeros_like(a_beta_metric) # 无法得知deltaf的导数如何求得，暂时先设置为0，看看控制效果再说

        return a_beta_metric, a_dbeta_metric, a_ddbeta_metric


    def calc_Psid_dPsid_ddPsid_metric(self, aS_agent_states: AgentState, oS_obst_states: ObstState, ad_action: Action,
                                      a_Yd: jnp.ndarray, a_dYd: jnp.ndarray, a_ddYd: jnp.ndarray, a_dddYd: jnp.ndarray
                                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        a_Y, a_dY, a_ddY = self.calc_Y_dY_ddY(aS_agent_states, ad_action)
        a_Ye = a_Y - a_Yd
        a_sgn_Ye = jnp.sign(a_Ye)
        a_dYe = a_dY - a_dYd
        a_ddYe = a_ddY - a_ddYd

        a_beta, a_dbeta, a_ddbeta = self.calc_beta_dbeta_ddbeta_metric(aS_agent_states, ad_action) # metric

        a_vx = aS_agent_states[:, 2] / 3.6 # metric

        ao_BD, ao_dBD, ao_ddBD = self.calc_BD_dBD_ddBD(aS_agent_states, oS_obst_states)
        a_BD_sum = ao_BD.sum(axis=1)
        a_dBD_sum = ao_dBD.sum(axis=1)
        a_ddBD_sum = ao_ddBD.sum(axis=1)
        ao_BD_int = self.integrate_BD_update(ao_BD)
        a_BD_int_sum = ao_BD_int.sum(axis=1)

        a_Psid_metric = self.k1*jnp.exp(-self.v*a_BD_sum)*a_sgn_Ye/a_vx + self.k2*a_BD_sum*a_Ye/((a_BD_int_sum+1)*a_vx) \
            +a_dYd/a_vx - a_beta
        a_dPsid_metric = -self.k1 * self.v * jnp.exp(-self.v*a_BD_sum) * a_dBD_sum * a_sgn_Ye / a_vx \
            +self.k2 * (a_dBD_sum * a_Ye + a_BD_sum * a_dYe) / ((a_BD_int_sum + 1) * a_vx) \
            -self.k2 * a_BD_sum**2 * a_Ye / ((a_BD_int_sum + 1)**2 * a_vx) \
            +a_ddYd / a_vx - a_dbeta
        a_ddPsid_metric = self.k1 * self.v * jnp.exp(-self.v*a_BD_sum) * (self.v*a_dBD_sum**2-a_ddBD_sum) * a_sgn_Ye / a_vx \
            +self.k2 * (a_ddBD_sum*a_Ye+2*a_dBD_sum*a_dYe+a_BD_sum*a_ddYe) / ((a_BD_int_sum + 1) * a_vx) \
            -self.k2 * (3*a_BD_sum*a_dBD_sum*a_Ye+2*a_BD_sum**2*a_Ye) / ((a_BD_int_sum + 1)**2 * a_vx) \
            +2 * self.k2 * a_BD_sum**3 * a_Ye / ((a_BD_int_sum + 1)**3 * a_vx)\
            +a_dddYd / a_vx - a_ddbeta

        return a_Psid_metric, a_dPsid_metric, a_ddPsid_metric


    def calc_omegat(self):
        return 0 # 暂时将噪声置为0


    def calc_deltaf(self, aS_agent_states: jnp.ndarray, oS_obst_states: jnp.ndarray, ad_action: jnp.ndarray) -> jnp.ndarray:
        a_Psid_metric, a_dPsid_metric, a_ddPsid_metric = self.calc_Psid_dPsid_ddPsid_metric(aS_agent_states, oS_obst_states, ad_action)
        a_Psi_metric = aS_agent_states[:, 4] * jnp.pi/180 # metric
        a_dPsi_metric = a_gamma_metric = aS_agent_states[:, 5] * jnp.pi/180 # metric
        a_e1 = a_Psi_metric - a_Psid_metric
        a_e2 = a_dPsi_metric - a_dPsid_metric
        a_z1 = a_e1 / self.Delta1
        a_z1_p = nthroot(a_z1**self.p_num, self.p_den)
        a_z2 = a_e2 / self.Delta2

        a_s = jnp.log((1+a_z2)/(1-a_z2)) + jnp.log((1+a_z1_p)/(1-a_z1_p))

        a_vx_metric = aS_agent_states[:, 2] / 3.6 # metric
        a_beta_metric, _, _ = self.calc_beta_dbeta_ddbeta_metric(aS_agent_states, ad_action)  # metric
        a_f = self.Lf * self.Cf * (a_beta_metric + a_gamma_metric * self.Lf / a_vx_metric) / self.Iz \
            +self.Lr * self.Cr * (-a_beta_metric + a_gamma_metric * self.Lr / a_vx_metric) / self.Iz
        a_b = -self.Lf * self.Cf / self.Iz

        omegat = self.calc_omegat()
        a_Gamma1 = 2 / (self.Delta2 * (1 - a_z2**2))
        a_Gamma2 = 2 * (a_f + omegat - a_ddPsid_metric) / (self.Delta2 * (1 - a_z2**2)) \
            +2 * self.p_num/self.p_den * a_z1_p/a_z1 * a_e2 / (self.Delta1 * (1 - a_z1_p**2))

        a_deltaf_metric = (-self.k3 * jnp.abs(a_s)**self.alpha * jnp.sign(a_s) - self.k4 * a_s - a_Gamma2) / (a_Gamma1 * a_b)
        a_deltaf_clip = jnp.clip(a_deltaf_metric * 180/jnp.pi, -7., 7.) # 限制在±7°之内

        return a_deltaf_clip