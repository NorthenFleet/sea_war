# hexapod_env.py

import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import time


class HexapodEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False):
        super(HexapodEnv, self).__init__()

        # 物理引擎是否渲染
        self.render_mode = render
        if self.render_mode:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # 加载地面
        self.plane = p.loadURDF("plane.urdf")

        # 加载 Hexapod 机器人
        self.robot_urdf = os.path.join(
            os.path.dirname(__file__), "assets", "hexapod.urdf")
        self.start_pos = [0, 0, 0.2]
        self.start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot = p.loadURDF(
            self.robot_urdf, self.start_pos, self.start_orientation)

        # 获取机器人的关节信息
        self.num_joints = p.getNumJoints(self.robot)
        self.joint_indices = list(range(self.num_joints))

        # 动作空间：每个关节的角度控制（假设每个关节有两个自由度）
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)

        # 状态空间：机器人关节的位置和速度
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_joints * 2,), dtype=np.float32)

        # 重置环境
        self.reset()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(
            self.robot_urdf, self.start_pos, self.start_orientation)

        # 初始化关节状态
        for joint in self.joint_indices:
            p.resetJointState(self.robot, joint,
                              targetValue=0.0, targetVelocity=0.0)

        self.prev_position = p.getBasePositionAndOrientation(self.robot)[
            0][0:2]  # x, y 位置
        state = self._get_obs()
        return state

    def step(self, action):
        # 应用动作
        for i, joint in enumerate(self.joint_indices):
            p.setJointMotorControl2(bodyUniqueId=self.robot,
                                    jointIndex=joint,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=action[i])

        p.stepSimulation()
        time.sleep(1./240.)  # 模拟真实时间速度

        # 获取新状态
        state = self._get_obs()

        # 计算奖励（前进的距离）
        current_position = p.getBasePositionAndOrientation(self.robot)[0][0:2]
        distance = np.linalg.norm(
            np.array(current_position) - np.array(self.prev_position))
        reward = distance  # 简单地将前进的距离作为奖励
        self.prev_position = current_position

        # 检查是否结束
        done = False
        if p.getBasePositionAndOrientation(self.robot)[0][2] < 0.1:  # 机器人跌落
            done = True
            reward = -100  # 跌落惩罚

        info = {}
        return state, reward, done, info

    def render(self, mode='human'):
        pass  # 已在初始化中处理

    def close(self):
        p.disconnect()

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot, self.joint_indices)
        obs = []
        for state in joint_states:
            obs.append(state[0])  # 位置
            obs.append(state[1])  # 速度
        return np.array(obs, dtype=np.float32)
