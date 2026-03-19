import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics


class NavigateAviary(BaseRLAviary):
    
    """Single-agent navigation: random start -> avoid circular obstacles -> random goal."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = False,
        record: bool = False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.VEL,
        map_size: float = 8.0,       # ( 8x8)
        grid_bins: int = 5,
        min_clearance: float = 0.3,
        goal_obstacle_margin: float = 1.0,
        start_goal_min_dist: float = 3.0,
    ):
        self.EPISODE_LEN_SEC = 20    # (600步)
        self.MAP_SIZE = float(map_size)
        self.GRID_BINS = int(grid_bins)
        self.MIN_CLEARANCE = float(min_clearance)
        self.GOAL_OBSTACLE_MARGIN = float(goal_obstacle_margin)
        self.START_GOAL_MIN_DIST = float(start_goal_min_dist)

        self.goal_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.start_pos = np.array([-3.0, -3.0, 1.0], dtype=np.float32)
        self.obstacles = []
        self.np_random = np.random.RandomState(0)

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )
        
        # 提高无人机的最大速度限制 (默认 BaseRLAviary 是 max_speed 的 3%。这里我们改为 15%，大约 1.2 m/s)
        if self.ACT_TYPE == ActionType.VEL:
            self.SPEED_LIMIT = 0.15 * self.MAX_SPEED_KMH * (1000/3600)

    def reset(self, seed: int = None, options: dict = None):
        if seed is not None:
            self.np_random = np.random.RandomState(int(seed))
        return super().reset(seed=seed, options=options)

    def generate_obstacles_grid(self):
        """Generate one circular obstacle per grid cell with anti-overlap checks."""
        self.obstacles = []
        half = self.MAP_SIZE / 2.0
        grid_size = self.MAP_SIZE / float(self.GRID_BINS)

        for gx in range(self.GRID_BINS):
            for gy in range(self.GRID_BINS):
                cell_cx = -half + (gx + 0.5) * grid_size
                cell_cy = -half + (gy + 0.5) * grid_size
                placed = False
                for _ in range(50):
                    # 把障碍物的半径缩小，设为 0.1 到 0.25 (即直径 20~50 厘米，类似树干或柱子)
                    radius = float(self.np_random.uniform(0.1, 0.25))
                    jitter = 0.28 * grid_size
                    ox = float(cell_cx + self.np_random.uniform(-jitter, jitter))
                    oy = float(cell_cy + self.np_random.uniform(-jitter, jitter))

                    if abs(ox) > half - (radius + self.MIN_CLEARANCE):
                        continue
                    if abs(oy) > half - (radius + self.MIN_CLEARANCE):
                        continue

                    overlap = False
                    for px, py, pr in self.obstacles:
                        d = float(np.hypot(ox - px, oy - py))
                        if d < (radius + pr + self.MIN_CLEARANCE):
                            overlap = True
                            break
                    if overlap:
                        continue

                    self.obstacles.append((ox, oy, radius))
                    placed = True
                    break

                if not placed:
                    continue

    def sample_free_goal(self):
        """Sample a goal not inside/too-close to obstacles (margin r + 1.5 by default)."""
        half = self.MAP_SIZE / 2.0
        for _ in range(500):
            gx = float(self.np_random.uniform(-half + 0.8, half - 0.8))
            gy = float(self.np_random.uniform(-half + 0.8, half - 0.8))
            safe = True
            for ox, oy, radius in self.obstacles:
                if np.hypot(gx - ox, gy - oy) < (radius + self.GOAL_OBSTACLE_MARGIN):
                    safe = False
                    break
            if safe:
                self.goal_pos = np.array([gx, gy, 1.0], dtype=np.float32)
                return
        self.goal_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def sample_free_start(self):
        """Sample start outside obstacles and farther than start_goal_min_dist from goal."""
        half = self.MAP_SIZE / 2.0
        for _ in range(500):
            sx = float(self.np_random.uniform(-half + 0.8, half - 0.8))
            sy = float(self.np_random.uniform(-half + 0.8, half - 0.8))

            if np.hypot(sx - self.goal_pos[0], sy - self.goal_pos[1]) < self.START_GOAL_MIN_DIST:
                continue

            safe = True
            for ox, oy, radius in self.obstacles:
                if np.hypot(sx - ox, sy - oy) < (radius + self.MIN_CLEARANCE):
                    safe = False
                    break
            if safe:
                self.start_pos = np.array([sx, sy, 1.0], dtype=np.float32)
                return
        self.start_pos = np.array([-half + 1.0, -half + 1.0, 1.0], dtype=np.float32)

    def _sample_layout(self):
        self.generate_obstacles_grid()
        self.sample_free_goal()
        self.sample_free_start()

    def _observationSpace(self):
        obs_space = super()._observationSpace()
        low = obs_space.low
        high = obs_space.high
        #  12 维: 相对目标(3维) + 3个最近障碍物特征(每个占3维: dx, dy, radius) = 12维
        extra_low = -np.inf * np.ones((self.NUM_DRONES, 12))
        extra_high = np.inf * np.ones((self.NUM_DRONES, 12))
        return spaces.Box(
            low=np.hstack([low, extra_low]),
            high=np.hstack([high, extra_high]),
            dtype=np.float32,
        )

    def _computeObs(self):
        obs = super()._computeObs()
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rel_goal = self.goal_pos - pos

        obs_dists = []
        for ox, oy, radius in self.obstacles:
            dx = ox - pos[0]
            dy = oy - pos[1]
            boundary_dist = float(np.hypot(dx, dy) - radius)
            # 只能观测到 3m 以内的障碍物
            if boundary_dist <= 3.0:
                obs_dists.append((boundary_dist, dx, dy, radius))

        obs_dists.sort(key=lambda x: x[0])  # 按距离从小到大排序
        
        nearest_obs_list = []
        for i in range(3):
            # 取最近的3个，如果3米内没有那么多障碍物，用非常大的安全假特征（比如距离无穷远，半径0）来填充
            if i < len(obs_dists):
                nearest_obs_list.extend([obs_dists[i][1], obs_dists[i][2], obs_dists[i][3]])
            else:
                nearest_obs_list.extend([10.0, 10.0, 0.0]) # 表示很远且绝对安全的假目标

        nearest_array = np.array(nearest_obs_list, dtype=np.float32)
        # 将原观测数据、目标相对向量、最近障碍物信息拼接在一起，喂给神经网络
        extended = np.hstack([obs[0], rel_goal, nearest_array]).astype(np.float32)
        return np.array([extended], dtype=np.float32)

    def _housekeeping(self):
        self._sample_layout()
        if self.INIT_XYZS is None or np.array(self.INIT_XYZS).shape != (1, 3):
            self.INIT_XYZS = np.zeros((1, 3), dtype=np.float32)
        self.INIT_XYZS[0] = self.start_pos
        self.last_vel = np.zeros(3, dtype=np.float32)
        # 初始化上一帧的距离
        self.last_dist_goal = float(np.linalg.norm(self.goal_pos - self.start_pos))
        super()._housekeeping()

        goal_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.2,
            rgbaColor=[0.1, 0.9, 0.1, 0.9],
            physicsClientId=self.CLIENT,
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=goal_visual,
            basePosition=self.goal_pos.tolist(),
            physicsClientId=self.CLIENT,
        )

    def _addObstacles(self):
        for ox, oy, radius in self.obstacles:
            col = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=radius,
                height=2.8,
                physicsClientId=self.CLIENT,
            )
            vis = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=radius,
                length=2.8,
                rgbaColor=[0.9, 0.25, 0.25, 0.85],
                physicsClientId=self.CLIENT,
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[ox, oy, 1.4],
                physicsClientId=self.CLIENT,
            )

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]

        # 1. 向目标靠近的势能奖励 
        dist_goal = float(np.linalg.norm(self.goal_pos - pos))
        if not hasattr(self, 'last_dist_goal'):
            self.last_dist_goal = dist_goal
        
        # progress为正说明在靠近，为负说明在逃离(quansu大概1.2)
        progress_reward = (self.last_dist_goal - dist_goal) * 30.0 
        self.last_dist_goal = dist_goal

        # 2. 轻微的时间惩罚 
        time_penalty = -0.05

        # 3. 静态安全奖励 (Rstatic)
        # 用距离各障碍物表面以及边界(墙壁)的距离近似模拟 D_lidar
        dists = []
        half = self.MAP_SIZE / 2.0
        dists.append(half - pos[0])
        dists.append(pos[0] - (-half))
        dists.append(half - pos[1])
        dists.append(pos[1] - (-half))
        for ox, oy, radius in self.obstacles:
            boundary_dist = float(np.hypot(pos[0] - ox, pos[1] - oy) - radius)
            dists.append(boundary_dist)
        
        # 只取距离最近的3个危险源
        dists.sort()
        closest_3_dists = dists[:3]
        dists_clipped = np.clip(closest_3_dists, 1e-6, np.inf)
        
        Rstatic = float(np.mean(np.log(dists_clipped)))*0.5

        # 4. 动作平滑惩罚 (Psmooth)
        Psmooth = float(np.linalg.norm(vel - self.last_vel))
        
        # 5. 高度越界惩罚 (Pheight)
        # 起点 z=1.0, 终点 z=1.0, 构成高度区间 [1.0, 1.0]. 宽容裕度 0.2m
        min_allowed_z = min(self.start_pos[2], self.goal_pos[2]) - 0.2
        max_allowed_z = max(self.start_pos[2], self.goal_pos[2]) + 0.2
        if pos[2] < min_allowed_z:
            Pheight = (min_allowed_z - pos[2]) ** 2
        elif pos[2] > max_allowed_z:
            Pheight = (pos[2] - max_allowed_z) ** 2
        else:
            Pheight = 0.0

        self.last_vel = vel.copy()

        # 综合日常奖励
        reward = progress_reward + time_penalty + Rstatic - Psmooth - 8.0 * Pheight
        
        # 碰到障碍物或到终点的奖励判断
        if dist_goal < 0.45:
            reward += 1000.0

        for ox, oy, radius in self.obstacles:
            boundary_dist = float(np.hypot(pos[0] - ox, pos[1] - oy) - radius)
            if boundary_dist < 0.05:
                reward -= 500.0
                break

        # 碰壁以及极小高度判定(掉落地面)
        half = self.MAP_SIZE / 2.0
        if abs(pos[0]) > half or abs(pos[1]) > half:
            reward -= 500.0  # 飞出边界视为和撞墙一样严重的自杀
        if pos[2] < 0.15:
            reward -= 500.0  # 坠机(掉到地上)也给严重惩罚
            
        return float(reward)

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        if np.linalg.norm(self.goal_pos - pos) < 0.45:
            return True

        for ox, oy, radius in self.obstacles:
            if np.hypot(pos[0] - ox, pos[1] - oy) < (radius + 0.05):
                return True
                
        # 边界与坠机作为 Terminated (直接结束)
        half = self.MAP_SIZE / 2.0
        if abs(pos[0]) > half or abs(pos[1]) > half:
            return True
        if pos[2] < 0.15 or pos[2] > 2.8:
            return True
            
        return False

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        is_success = np.linalg.norm(self.goal_pos - pos) < 0.45
        return {
            "goal": self.goal_pos.copy(),
            "start": self.start_pos.copy(),
            "obstacles": list(self.obstacles),
            "is_success": is_success,
        }
