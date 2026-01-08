# 状态量
SingleObjectState[] aS_agent_states # ego
SingleObjectState[] aS_goal_states  # goal
SingleObjectState[] oS_obst_states  # obstacles
int32 state_dim # state_dim=8，方便解析

# 环境step时刻的动作量，方便后续统计和作图
SingleAgentControl[] ad_actions
int32 action_dim # action_dim=2, 方便解析

# 评价量
float32 reward # -l(xk)
float32 cost # max{h(xk)}
float32 reward_max # env中的参数
float32 reward_min # env中的参数
