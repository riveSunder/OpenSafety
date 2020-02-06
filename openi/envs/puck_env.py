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
        cube_position, cube_orientation = p.getBasePositionAndOrientation(self.bot_id)

        cube_orientation = p.getEulerFromQuaternion(cube_orientation)
        v = p.getBaseVelocity(self.bot_id, self.physicsClient)
        print(cube_position, cube_orientation)

        
        obs = v
        return obs

    def compute_force(self, action):

        cube_position, cube_orientation = p.getBasePositionAndOrientation(self.bot_id)

        cube_orientation = p.getEulerFromQuaternion(cube_orientation)
        

        force = [action[0] * np.cos(cube_orientation[2]), action[1] * np.sin(cube_orientation), 0]

    def step(self, action):
        
        #p.resetBaseVelocity(self.bot_id, linearVelocity=[action[0], 0, 0], \
        #        angularVelocity=[0, 0, action[1]])

        force = self.compute_force(action)
        p.applyExternalForce(self.bot_id, -1, [action[0],0,0], [0,0,0.], \
                flags=p.LINK_FRAME, physicsClientId=self.physicsClient)
        p.applyExternalTorque(self.bot_id, -1, [0,0,action[1]], \
                flags=p.LINK_FRAME, physicsClientId=self.physicsClient)

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
    info = p.getDynamicsInfo(env.bot_id, -1)
    print(info)
    p.changeDynamics(env.bot_id,-1, lateralFriction=0.01)
    info = p.getDynamicsInfo(env.bot_id, -1)
    print(info)
    
    for ii in range(500):
        time.sleep(0.025)
        p.stepSimulation()
        action = np.random.randn(2)
        #action = np.array([0, 5])
        obs, reward, done, info = env.step(action)
        
