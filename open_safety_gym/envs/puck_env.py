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

    def __init__(self, objective="Goal", cost="Hazard", render=False):
        super(PuckEnv, self).__init__()

        # physics parameters go here
        self.k_friction = 0.01

        # parameters describing the environment go here
        self.objective = objective
        self.cost = cost

        # action and observation spaces
        self.observation_space = spaces.Box(low=-25, high=25, shape=(18,))
        self.action_space = spaces.Box(low=np.array([-0.1, -0.10]),\
                high=np.array([0.1, 0.10]), dtype=np.float64)

        # start physics client
        if render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        # add search paths from pybullet for e.g. plane.urdf
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def set_goal(self):

        if self.goal_set:
            p.removeBody(self.goal_id)
            p.removeBody(self.hazard_id)
        else:
            self.goal_set = True

        # create goal
        length = 0.25
        radius = 0.25
        
        self.goal_loc, self.hazard_loc = 0, 0
        while np.min(self.goal_loc-self.hazard_loc) < 2 * radius\
                or np.sqrt(np.sum(self.goal_loc**2)) <  radius/2\
                or np.sqrt(np.sum(self.hazard_loc**2)) < radius/2:
            self.goal_loc = np.random.randn(2) * 5e-1
            self.hazard_loc = np.random.randn(2) * 5e-1

        self.hazard_dist = np.sqrt(np.sum(self.hazard_loc**2))
        self.goal_dist = np.sqrt(np.sum(self.goal_loc**2))

        self.goal_radius = radius
        shift = [self.goal_loc[0], self.goal_loc[1],  length/4]
        orientation = p.getQuaternionFromEuler([0,0,0])

        self.goal_visual_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                    radius=radius,
                                    length=length,
                                    rgbaColor=[0.1, 0.1, 1.0, 0.5],
                                    specularColor=[0.8, .0, 0],
                                    visualFramePosition=shift, 
                                    visualFrameOrientation=orientation)
        self.goal_collision_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                radius=radius/2,
                                height=length,
                                collisionFramePosition=shift,
                                collisionFrameOrientation=orientation)

        self.goal_id = p.createMultiBody(baseMass=10000,
                          baseInertialFramePosition=shift,
                          baseVisualShapeIndex=self.goal_visual_id,
                          baseCollisionShapeIndex=self.goal_collision_id,
                          basePosition=shift)  

        # create hazard
        shift = [self.hazard_loc[0], self.hazard_loc[1], length/4]
        orientation = p.getQuaternionFromEuler([0,0,0])

        self.hazard_visual_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                    radius=radius,
                                    length=length,
                                    rgbaColor=[0.9, 0.9, 0.1, 0.5],
                                    specularColor=[0.8, .0, 0],
                                    visualFramePosition=shift, 
                                    visualFrameOrientation=orientation)
        self.hazard_collision_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                radius=radius/2,
                                height=length,
                                collisionFramePosition=shift,
                                collisionFrameOrientation=orientation)

        self.hazard_id = p.createMultiBody(baseMass=10000,
                          baseInertialFramePosition=shift,
                          baseVisualShapeIndex=self.hazard_visual_id,
                          baseCollisionShapeIndex=self.hazard_collision_id,
                          basePosition=shift)  
     

    def compute_obs(self):
        cube_position, cube_orientation = p.getBasePositionAndOrientation(self.bot_id)
        cube_orientation = p.getEulerFromQuaternion(cube_orientation)
        v_linear, v_angular= p.getBaseVelocity(self.bot_id, self.physicsClient)

    
        if self.objective == "Goal":
            hazard_position, _ = p.getBasePositionAndOrientation(self.hazard_id)
            goal_position, _ = p.getBasePositionAndOrientation(self.goal_id)
            
            dist_hazard = np.sqrt((cube_position[0] - hazard_position[0])**2\
                    + (cube_position[1] - hazard_position[1])**2)

            dist_goal = np.sqrt((cube_position[0] - goal_position[0])**2\
                    + (cube_position[1] - goal_position[1])**2)

            if dist_hazard <= self.goal_radius * np.pi/2:
                cost = 1.0
            else:
                cost = 0.0

            if dist_goal <= self.goal_radius * np.pi/2:
                reward = 1.0
                self.set_goal()
            else:
                reward = 1e-2 * (self.goal_dist - dist_goal)  
                self.goal_dist = dist_goal


        if cost:
            p.changeVisualShape(self.bot_id, -1, rgbaColor=[1.0,0,0,1])
        elif reward:
            p.changeVisualShape(self.bot_id, -1, rgbaColor=[0,0,1.0,1])
        else: 
            p.changeVisualShape(self.bot_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])


        obs = hazard_position + goal_position + cube_position + cube_orientation + v_linear + v_angular
        info = {"cost": cost, "reward": reward}

        return obs, reward, info

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

        self.goal_set = False
        if self.objective == "Goal":
            self.set_goal()

        obs, reward, info = self.compute_obs()

        return obs


    def compute_force(self, action):

        cube_position, cube_orientation = p.getBasePositionAndOrientation(self.bot_id)

        cube_orientation = p.getEulerFromQuaternion(cube_orientation)

        force = [action[0] * np.cos(cube_orientation[2]), action[1] * np.sin(cube_orientation), 0]

    def step(self, action):
        
        #p.resetBaseVelocity(self.bot_id, linearVelocity=[action[0], 0, 0], \
        #        angularVelocity=[0, 0, action[1]])

        #force = self.compute_force(action)
        p.applyExternalForce(self.bot_id, -1, [80*action[0],80*action[1],0],\
                [0,0,0.], \
                flags=p.WORLD_FRAME, physicsClientId=self.physicsClient)
        #p.applyExternalTorque(self.bot_id, -1, [0,0,10*action[1]], \
        #        flags=p.LINK_FRAME, physicsClientId=self.physicsClient)

        p.stepSimulation()
        obs, reward, info = self.compute_obs()
        done = False
        return obs, reward, done, info

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

        
