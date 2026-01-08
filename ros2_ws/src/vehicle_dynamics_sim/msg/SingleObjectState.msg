# 单个车辆（ego、goal或obstacle）的在世界坐标系下的状态量
float32 x            # x方向位置 m
float32 y            # y方向位置 m
float32 vx           # x方向速度 km/h
float32 vy           # y方向速度 km/h
float32 theta        # 航向角 deg
float32 dthetadt     # 航向角角速度 deg/s
float32 bw           # 包围盒宽度（纵向） m
float32 bh           # 包围盒长度（横向） m
