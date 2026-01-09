# 状态量
SingleObjectState[] aS_agent_states # ego
SingleObjectState[] aS_goal_states  # goal
SingleObjectState[] oS_obst_states  # obstacles
int32 state_dim # state_dim=8，方便解析

# 评价量
float32 reward # -l(xk)
float32 cost # max{h(xk)}
