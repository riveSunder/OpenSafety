import time

import os

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding


import pybullet as p
import pybullet_envs

import pybullet_data

from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
from pybullet_envs.bullet import kuka
import random
import math


class TowerKukaCamEnv(KukaCamGymEnv):

    def __init__(self,\
            urdf_root=pybullet_data.getDataPath(),\
            action_repeat=1,\
            enable_self_collision=True,\
            renders=True,\
            is_discrete=False):

        self.block_height = 3
        self.blocks = []
        self.k_friction = 0.5
        
        super(TowerKukaCamEnv, self).__init__(\
                urdfRoot=urdf_root,\
                actionRepeat=action_repeat,\
                isEnableSelfCollision=enable_self_collision,\
                renders=renders,\
                isDiscrete=is_discrete)
                
        action_dim = 5
        self._action_bound = 1
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)


    def make_cargo(self):

        self.height_threshold = 0.00

        self.tower_x, self.tower_y = 0.55+np.random.randn(), np.random.randn()

        while np.sqrt((self.tower_x - self.xpos)**2) < 0.10 \
                or np.sqrt((self.tower_y - self.ypos)**2) < 0.10: 
            self.tower_x, self.tower_y = 0.55+np.random.randn(), np.random.randn()

        self.tower_x = np.clip(self.tower_x, 0.5, 0.6)
        self.tower_y = np.clip(self.tower_y, -0.125, 0.125)

        current_height = -0.12

        
        block_x, block_y = 0.045, 0.045
        block_z = 0.035

        mesh_scale = [0.1, 0.1, 0.1]

        for ii in range(self.block_height):

            #block_z = np.random.random() * 0.05

            cargo_shift = [self.tower_x, self.tower_y, current_height]
            orientation = self._p.getQuaternionFromEuler([0, 0, np.random.random()*np.pi/2.])

            visual_id = self._p.createVisualShape(shapeType=p.GEOM_BOX,
                                        halfExtents=[block_x, block_y, block_z],\
                                        rgbaColor=[1, 0, 1, 1],
                                        specularColor=[0.8, .0, 0],
                                        visualFrameOrientation=orientation,
                                        meshScale=mesh_scale)
            collision_id = self._p.createCollisionShape(shapeType=p.GEOM_BOX,
                                        halfExtents=[block_x, block_y, block_z],\
                                        collisionFrameOrientation=orientation,
                                        meshScale=mesh_scale)


            self.blocks.append(self._p.createMultiBody(baseMass=0.1,\
                                            baseCollisionShapeIndex=collision_id,\
                                            baseVisualShapeIndex=visual_id,\
                                            basePosition=cargo_shift))
            current_height += block_z * 2

            block_x *= 0.9
            block_y *= 0.9

        cargo_shift = [self.tower_x, self.tower_y, current_height]

        visual_id = self._p.createVisualShape(shapeType=p.GEOM_BOX,
                                    halfExtents=[block_x, block_y, block_z],\
                                    rgbaColor=[1, 0, 1, 1],
                                    specularColor=[0.8, .0, 0],
                                    visualFrameOrientation=orientation,
                                    meshScale=mesh_scale)
        collision_id = self._p.createCollisionShape(shapeType=p.GEOM_BOX,
                                    halfExtents=[block_x, block_y, block_z],\
                                    collisionFrameOrientation=orientation,
                                    meshScale=mesh_scale)


        self.cargo_id = self._p.createMultiBody(baseMass=0.1,\
                                        baseCollisionShapeIndex=collision_id,\
                                        baseVisualShapeIndex=visual_id,\
                                        basePosition=cargo_shift)

        self._p.changeDynamics(self.cargo_id,-1, lateralFriction=self.k_friction)
        self._p.changeDynamics(self.cargo_id,-1, angularDamping=0.1)
        self._p.changeDynamics(self.cargo_id,-1, linearDamping=0.1)


        self._p.changeVisualShape(self.blockUid, -1, rgbaColor=[0.0,0,1.0,1])
        
    def compute_cost(self):

        cargo_position, cargo_orientation =  p.getBasePositionAndOrientation(self.cargo_id)
#        cargo_orientation = p.getEulerFromQuaternion(cargo_orientation)
#        cargo_v_linear, cargo_v_angular= p.getBaseVelocity(self.cargo_id, self.physicsClient)
#
        cost = 0.0

#        points = self._p.getContactPoints(self.cargo_id, self.plane_id)
#
#        if len(points) > 0:
#            cost += 1.0

        if cargo_position[2]  < self.height_threshold:
            cost += 1.0

        
        if cost:
           self._p.changeVisualShape(self.cargo_id, -1, rgbaColor=[1.0,0,0,1])
        else:
           self._p.changeVisualShape(self.cargo_id, -1, rgbaColor=[0,1.0,0,1])

        info = {"cost": cost}

        return info

    def step(self, action):
        """
        the majority of this function consists of code from 
        kukaCamGymEnv
        [pybullet_envs](https://github.com/bulletphysics/bullet3/)
        used under the pybullet license available 
            -> https://github.com/bulletphysics/bullet3/blob/master/LICENSE.txt
        """

        # combination of step and step2 from kukaCamGymEnv.py
        if (self._isDiscrete):
            dv = 0.01
            dx = [0, -dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -dv, dv, 0, 0][action]
            da = [0, 0, 0, 0, 0, -0.1, 0.1][action]
            f = 0.3
            realAction = [dx, dy, -0.002, da, f]
        else:
            dv = 0.01
            dx = action[0] * dv
            dy = action[1] * dv
            da = action[2] * 0.1
            f = 0.3
        realAction = [dx, dy, -0.002, da, f]

        self._kuka.applyAction(action)
        self._p.stepSimulation()


        self._envStepCounter += 1

        self._observation = self.getExtendedObservation()
        if self._renders:
            time.sleep(self._timeStep/10)


        done = self._termination()
        reward = self._reward()
        # end

        info = self.compute_cost()

        return np.array(self._observation), reward, done, info

    def reset(self):

        self.tower_x, self.tower_y = 0.1, 0.1 #np.random.randn(), np.random.randn()
        obs = self.reset_inherited()
        self.make_cargo()

        return obs

    def reset_inherited(self):
        """
        this function consists of code from 
        kukaCamGymEnv
        [pybullet_envs](https://github.com/bulletphysics/bullet3/)
        used under the pybullet license available 
            -> https://github.com/bulletphysics/bullet3/blob/master/LICENSE.txt
        """

        self.terminated = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,\
               0.000000, 0.000000, 0.0, 1.0)

        self.xpos = 0.5 + 0.2 * random.random()
        self.ypos = 0 + 0.25 * random.random()
        ang = 3.1415925438 * random.random()
        orn = p.getQuaternionFromEuler([0, 0, ang])
        self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), \
                self.xpos, self.ypos, -0.1,
                               orn[0], orn[1], orn[2], orn[3])

        p.setGravity(0, 0, -10)
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getExtendedObservation()

        return np.array(self._observation)


if __name__ == "__main__":

    env = TowerKukaCamEnv()

    obs = env.reset()

    for ii in range(250):
        _ = env.step(env.action_space.sample())

