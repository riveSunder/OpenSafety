import time
import os

import gym
from gym import error, spaces, utils
from gym.utils import seeding


import pybullet
import pybullet_data


class MezzaLunaEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render=False):
        super(MezzaLunaEnv, self).__init__()

        if render:
            self.physics_client = pybullet.connect(pybullet.GUI)
        else:
            self.physics_client = pybullet.connect(pybullet.DIRECT)

    def step(self, action):
        pass

    def reset(self):
        pybullet.resetSimulation()
        pybullet.setGravity(0, 0, -10)
        pybullet.setTimeStep(0.01)
        plane_ID = pybullet.loadURDF("plane.urdf")
        cube_start_position = [0, 0, 0.001]
        cube_start_orientation = pybullet.getQuaternionFromEuler([0, 0, 0])
        path = os.path.abspath(os.path.dirname(__file__))   

        self.bot_id = pybullet.loadURDF(os.path.join(path, "mezza_luna.xml"),\
            cube_start_position,\
            cube_start_orientation)

        return 0

    def render(self, mode="human", close=False):
        pass



if __name__ == "__main__":

    env = MezzaLunaEnv(render=True)

    obs = env.reset()
    for ii in range(300):
        pybullet.stepSimulation()
        time.sleep(0.025)
        
