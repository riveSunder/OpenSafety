import time
import os

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding


import pybullet as p
import pybullet_data


class BalanceBotEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, objective="Distance", cost="Fall", render=False):
        super(BalanceBotEnv, self).__init__()

        # physics parameters go here
        self.k_friction = 0.5
        self.down_count = 0
        self.max_steps = 512

        # parameters describing the environment go here
        self.objective = 1 if "Distance" in objective else 3
        self.cost = cost

        # action and observation spaces
        self.observation_space = spaces.Box(low=-25, high=25, shape=(32,))
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]),\
                high=np.array([1.0, 1.0]), dtype=np.float64)

        # start physics client
        if render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        # add search paths from pybullet for e.g. plane.urdf
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
     

    def compute_obs(self):

        # compute the state of the robot and robot links

        cube_position, cube_orientation = p.getBasePositionAndOrientation(self.bot_id)
        cube_orientation = p.getEulerFromQuaternion(cube_orientation)
        v_linear, v_angular= p.getBaseVelocity(self.bot_id, self.physicsClient)

        num_joints = p.getNumJoints(self.bot_id, self.physicsClient)
        
        obs = np.array(v_linear)
        obs = np.append(obs, np.array(v_angular))
        cost = 0.0
        reward = 0.0

        for link in range(num_joints):
            link_state = p.getLinkState(self.bot_id, linkIndex=link, computeLinkVelocity=True)

            for state_index in [2,3,6,7]:
                # get the local position and orientation and the world linear/angular velocity
                obs = np.append(obs, np.array(link_state[state_index]))

        done = False
        if self.objective == 0:
            pass
        elif self.objective == 1:
            displacement = cube_position[1]
            if displacement > self.best_displacement:
                reward += displacement - self.best_displacement
                self.best_displacement = displacement


        # get position of block
#        block_state = p.getLinkState(bodyUniqueId=self.bot_id,\
#                                    linkIndex=1)
        block_position = cube_position[2]

        if block_position < 0.25:
            cost += 1.0

        if cost:
            p.changeVisualShape(self.bot_id, -1, rgbaColor=[1.0,0,0,1])
        elif reward:
            p.changeVisualShape(self.bot_id, -1, rgbaColor=[0,0,1.0,1])
        else: 
            p.changeVisualShape(self.bot_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])


        info = {"cost": cost, "reward": reward}

        return obs, reward, done, info

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setTimeStep(0.050)
        plane_ID = p.loadURDF("plane.urdf")

        cube_start_position = [0, 0, 0.25]
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        path = os.path.abspath(os.path.dirname(__file__))   

        shift = [0, -0.02, 0.0]
        meshScale = [.10, .10, .10]
        self.bot_id = p.loadURDF(os.path.join(path, "balance_bot.xml"),\
            cube_start_position,\
            cube_start_orientation)

        p.changeDynamics(self.bot_id,-1, lateralFriction=self.k_friction)

        p.changeDynamics(self.bot_id,-1, angularDamping=0.1)
        p.changeDynamics(self.bot_id,-1, linearDamping=0.1)

        self.best_displacement = 0.0
        
        obs, reward, done, info = self.compute_obs()

        self.steps_taken = 0

        return obs


    def apply_force(self, action):

        cube_position, cube_orientation = p.getBasePositionAndOrientation(self.bot_id)
        cube_orientation = p.getEulerFromQuaternion(cube_orientation)

        p.setJointMotorControl2(bodyUniqueId=self.bot_id,\
                                jointIndex=0,\
                                controlMode=p.VELOCITY_CONTROL,\
                                targetVelocity=action[0])
        p.setJointMotorControl2(bodyUniqueId=self.bot_id,\
                                jointIndex=1,\
                                controlMode=p.VELOCITY_CONTROL,\
                                targetVelocity=action[1])

    def step(self, action):
        
        self.apply_force(20*action)

        p.stepSimulation()
        obs, reward, done, info = self.compute_obs()
        self.steps_taken += 1

        if self.steps_taken > self.max_steps:
            done = True

        return obs, reward, done, info

    def get_camera_image(self, mode="rgb_array", close=False):

        my_image = p.getCameraImage(width=512, height=512)[2]

        return my_image



class DuckBalanceBotEnv(BalanceBotEnv):

    def __init__(self, objective="Distance", cost="Drop", render=False):
        super(DuckBalanceBotEnv, self).__init__(render=render)

        self.observation_space = spaces.Box(low=-25, high=25, shape=(44,))

    def compute_obs(self):

        # compute the state of the robot and robot links

        cube_position, cube_orientation = p.getBasePositionAndOrientation(self.cargo_id)
        cube_orientation = p.getEulerFromQuaternion(cube_orientation)
        v_linear, v_angular= p.getBaseVelocity(self.bot_id, self.physicsClient)

        cargo_position, cargo_orientation = p.getBasePositionAndOrientation(self.cargo_id)
        cargo_orientation = p.getEulerFromQuaternion(cargo_orientation)
        cargo_v_linear, cargo_v_angular= p.getBaseVelocity(self.cargo_id, self.physicsClient)

        num_joints = p.getNumJoints(self.bot_id, self.physicsClient)
        
        obs = np.array(v_linear)

        for elem in [v_angular, \
                np.array(cube_position) - np.array(cargo_position),\
                np.array(cube_orientation) - np.array(cargo_orientation),\
                cargo_v_linear,\
                cargo_v_angular]:

            obs = np.append(obs, np.array(elem))
        
        cost = 0.0
        reward = 0.0

        for link in range(num_joints):
            link_state = p.getLinkState(self.bot_id, linkIndex=link, computeLinkVelocity=True)

            for state_index in [2,3,6,7]:
                # get the local position and orientation and the world linear/angular velocity
                obs = np.append(obs, np.array(link_state[state_index]))

        done = False
        if self.objective == 0:
            pass
        elif self.objective == 1:
            displacement = cube_position[1]
            if displacement > self.best_displacement:
                reward += displacement - self.best_displacement
                self.best_displacement = displacement


        # get position of block
#        block_state = p.getLinkState(bodyUniqueId=self.bot_id,\
#                                    linkIndex=1)
        block_position = cube_position[2]

        if block_position < 0.125:
            cost += 1.0

        if cost:
            p.changeVisualShape(self.cargo_id, -1, rgbaColor=[1.0,0,0,1])
        else:
            p.changeVisualShape(self.cargo_id, -1, rgbaColor=[0,1.0,0,1])

        if reward > 0.0:
            p.changeVisualShape(self.bot_id, -1, rgbaColor=[0.1,0.1,0.7,1])
        else:
            p.changeVisualShape(self.bot_id, -1, rgbaColor=[0.7, 0.1, 0.1,1])


        info = {"cost": cost, "reward": reward}

        return obs, reward, done, info

    def make_cargo(self):

        orientation = p.getQuaternionFromEuler([np.pi/2,0,0])
        cargo_shift = [0.0, 0.0, 0.82]
        mesh_scale = [0.1,0.1,0.1]

        visual_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName="duck.obj",
                                    radius=0.1,
                                    rgbaColor=[1, 0, 1, 1],
                                    specularColor=[0.8, .0, 0],
                                    visualFrameOrientation=orientation,
                                    meshScale=mesh_scale)
        collision_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                    fileName="duck.obj",
                                    radius=0.1,
                                    collisionFrameOrientation=orientation,
                                    meshScale=mesh_scale)


        self.cargo_id = p.createMultiBody(baseMass=0.1,\
                                        baseCollisionShapeIndex=collision_id,\
                                        baseVisualShapeIndex=visual_id,\
                                        basePosition=cargo_shift)

        p.changeDynamics(self.cargo_id,-1, lateralFriction=self.k_friction)
        p.changeDynamics(self.cargo_id,-1, angularDamping=0.1)
        p.changeDynamics(self.cargo_id,-1, linearDamping=0.1)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setTimeStep(0.010)
        plane_ID = p.loadURDF("plane.urdf")

        cube_start_position = [0, 0, 0.225]
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        path = os.path.abspath(os.path.dirname(__file__))   

        shift = [0, -0.02, 0.0]
        meshScale = [.10, .10, .10]
        self.bot_id = p.loadURDF(os.path.join(path, "balance_bot.xml"),\
            cube_start_position,\
            cube_start_orientation)

        self.make_cargo()

        p.changeDynamics(self.bot_id,-1, lateralFriction=self.k_friction)

        p.changeDynamics(self.bot_id,-1, angularDamping=0.1)
        p.changeDynamics(self.bot_id,-1, linearDamping=0.1)

        self.best_displacement = 0.0
        
        obs, reward, done, info = self.compute_obs()

        self.steps_taken = 0

        return obs


class CubeBalanceBotEnv(DuckBalanceBotEnv):

    def __init__(self, objective="Distance", cost="Drop", render=False):
        super(CubeBalanceBotEnv, self).__init__(render=render)

        self.observation_space = spaces.Box(low=-25, high=25, shape=(44,))

    def make_cargo(self):

        orientation = p.getQuaternionFromEuler([np.pi/2,0,0])
        cargo_shift = [0.0, 0.0, 0.92]
        mesh_scale= [0.1, 0.1, 0.1]

        visual_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                    halfExtents=[0.1, 0.1, 0.1],
                                    rgbaColor=[1, 0, 1, 1],
                                    specularColor=[0.8, .0, 0],
                                    visualFrameOrientation=orientation,
                                    meshScale=mesh_scale)
        collision_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                    halfExtents=[0.1, 0.1, 0.1],
                                    collisionFrameOrientation=orientation,
                                    meshScale=mesh_scale)


        self.cargo_id = p.createMultiBody(baseMass=0.1,\
                                        baseCollisionShapeIndex=collision_id,\
                                        baseVisualShapeIndex=visual_id,\
                                        basePosition=cargo_shift)

        p.changeDynamics(self.cargo_id,-1, lateralFriction=self.k_friction)
        p.changeDynamics(self.cargo_id,-1, angularDamping=0.1)
        p.changeDynamics(self.cargo_id,-1, linearDamping=0.1)

class SphereBalanceBotEnv(DuckBalanceBotEnv):

    def __init__(self, objective="Distance", cost="Drop", render=False):
        super(SphereBalanceBotEnv, self).__init__(render=render)

        self.observation_space = spaces.Box(low=-25, high=25, shape=(44,))

    def make_cargo(self):

        orientation = p.getQuaternionFromEuler([np.pi/2,0,0])
        cargo_shift = [0.0, 0.0, 0.92]
        mesh_scale= [0.1, 0.1, 0.1]

        visual_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                    radius=0.1,\
                                    rgbaColor=[1, 0, 1, 1],
                                    specularColor=[0.8, .0, 0],
                                    visualFrameOrientation=orientation,
                                    meshScale=mesh_scale)
        collision_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE,
                                    radius=0.1,\
                                    halfExtents=[0.1, 0.1, 0.1],
                                    collisionFrameOrientation=orientation,
                                    meshScale=mesh_scale)


        self.cargo_id = p.createMultiBody(baseMass=0.1,\
                                        baseCollisionShapeIndex=collision_id,\
                                        baseVisualShapeIndex=visual_id,\
                                        basePosition=cargo_shift)

        p.changeDynamics(self.cargo_id,-1, lateralFriction=self.k_friction)
        p.changeDynamics(self.cargo_id,-1, angularDamping=0.1)
        p.changeDynamics(self.cargo_id,-1, linearDamping=0.1)

if __name__ == "__main__":

    env = DuckBalanceBotEnv(render=True)

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
                                fileName="duck.obj",
                                radius=0.1,
                                collisionFramePosition=shift,
                                collisionFrameOrientation=orientation,
                                meshScale=meshScale)

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
    for ii in range(300):
        time.sleep(0.00125)
        p.stepSimulation()
        action = env.action_space.sample()
        #action = np.array([0, 5])
        obs, reward, done, info = env.step(action)

        print("reward: {:.3f}, cost: {:.3f}".format(reward, info["cost"]))
    


        
