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

    def __init__(self, objective="Goal", cost="Nothing", render=False):
        super(PuckEnv, self).__init__()

        # physics parameters go here
        self.k_friction = 0.01

        # parameters describing the environment go here
        self.objective = objective
        self.cost = cost

        # action and observation spaces
        self.observation_space = spaces.Box(low=np.array([-25., -25., -25., -np.pi, -np.pi, -np.pi, -25,-25,-25, -np.pi, -np.pi, -np.pi]),\
                high=np.array([25., 25., 25., np.pi, np.pi, np.pi, 25, 25, 25, np.pi, np.pi, np.pi]), dtype=np.float64)
        self.action_space = spaces.Box(low=np.array([-10.0, -1.0]), high=np.array([10.0, 1.0]), dtype=np.float64)

        # start physics client
        if render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        # add search paths from pybullet for e.g. plane.urdf
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def compute_obs(self):
        cube_position, cube_orientation = p.getBasePositionAndOrientation(self.bot_id)

        cube_orientation = p.getEulerFromQuaternion(cube_orientation)
        v_linear, v_angular= p.getBaseVelocity(self.bot_id, self.physicsClient)

        obs = cube_position + cube_orientation + v_linear + v_angular
        
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

        p.changeDynamics(self.bot_id,-1, lateralFriction=self.k_friction)
        return 0

    def render(self, mode="human", close=False):
        pass

if __name__ == "__main__":

    env = PuckEnv(render=True)

    obs = env.reset()
    info = p.getDynamicsInfo(env.bot_id, -1)

    shift = [-0.25,-0.25,-1]
    meshScale = [.1,.1,.1]
    orientation = p.getQuaternionFromEuler([np.pi/2,0,0])

    visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                fileName="duck.obj",
                                radius=0.1,
                                rgbaColor=[1, 1, 1, 1],
                                specularColor=[0.8, .0, 0],
                                visualFramePosition=shift,
                                visualFrameOrientation=orientation,
                                meshScale=meshScale)
    collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                fileName="duck_vhacd.obj",
                                radius=0.1,
                                collisionFramePosition=shift,
                                collisionFrameOrientation=orientation,
                                meshScale=meshScale)

#    visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE,
#                                radius=0.1,
#                                rgbaColor=[1, 1, 1, 1],
#                                specularColor=[0.8, .0, 0],
#                                visualFramePosition=shift,
#                                visualFrameOrientation=orientation,
#                                meshScale=meshScale)
#    collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_SPHERE,
#                                radius=0.1,
#                                collisionFramePosition=shift,
#                                collisionFrameOrientation=orientation,
#                                meshScale=meshScale)

    rangex = 1
    rangey = 1
    for i in range(rangex):
      for j in range(rangey):
        p.createMultiBody(baseMass=1,
                          baseInertialFramePosition=shift,
                          baseCollisionShapeIndex=collisionShapeId,
                          baseVisualShapeIndex=visualShapeId,
                          basePosition=[((-rangex / 2) + i) * meshScale[0] * 2,
                                        (-rangey / 2 + j) * meshScale[1] * 2, 1],
                              useMaximalCoordinates=False)

    time.sleep(2.)
    for ii in range(1000):
        time.sleep(0.025)
        p.stepSimulation()
        action = env.action_space.sample()
        #action = np.array([0, 5])
        obs, reward, done, info = env.step(action)

    p.createVisualShape(p.GEOM_SPHERE, radius=10, physicsClientId=env.physicsClient)
    obs, reward, done, info = env.step(action)
        
