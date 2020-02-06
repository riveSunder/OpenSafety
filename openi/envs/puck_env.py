import time
import os

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding


import pybullet as p
import pybullet_data


class PuckEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render=False):
        super(PuckEnv, self).__init__()

        if render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def compute_obs(self):
        v = p.getBaseVelocity(self.bot_id, self.physicsClient)
        print(v)

        obs = v
        return obs

    def step(self, action):
        
        p.resetBaseVelocity(self.bot_id, linearVelocity=[action[0], 0, 0], \
                angularVelocity=[0, 0, action[1]])

        p.stepSimulation()
        obs = self.compute_obs()
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setTimeStep(0.01)
        plane_ID = p.loadURDF("plane.urdf")
        cube_start_position = [0, 0, 0.001]
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        path = os.path.abspath(os.path.dirname(__file__))   

        shift = [0, -0.02, 0.0]
        meshScale = [.10, .10, .10]
        self.bot_id = p.loadURDF(os.path.join(path, "puck.xml"),\
            cube_start_position,\
            cube_start_orientation)
        return 0

    def render(self, mode="human", close=False):
        pass

if __name__ == "__main__":

    env = PuckEnv(render=True)

    obs = env.reset()
    for ii in range(500):
        time.sleep(0.025)
        p.stepSimulation()
        #action = np.random.randn(2)
        action = np.array([0.15, 5])
        obs, reward, done, info = env.step(action)
        
