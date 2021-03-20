import time
import os

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding


import pybullet as p
import pybullet_envs

import pybullet_data

from pybullet_envs.bullet.racecarZEDGymEnv import RacecarZEDGymEnv
from pybullet_envs.bullet import racecar
import random
import math

class SphereRacecarEnv(RacecarZEDGymEnv):

    def __init__(self,\
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=10,
               isEnableSelfCollision=True,
               isDiscrete=False,
               renders=True):

        self.k_friction = 0.5

        super(SphereRacecarEnv, self).__init__(urdfRoot=urdfRoot,\
                actionRepeat=actionRepeat,\
                isEnableSelfCollision=isEnableSelfCollision,\
                isDiscrete=isDiscrete,\
                renders=renders)

    def make_cargo(self):

        self.height_threshold = 0.20
        orientation = self._p.getQuaternionFromEuler([np.pi/2,0,0])
        cargo_shift = [0.15, 0.0, 0.28]
        mesh_scale = [0.1,0.1,0.1]

        visual_id = self._p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                    radius=0.1,\
                                    rgbaColor=[1, 0, 1, 1],
                                    specularColor=[0.8, .0, 0],
                                    visualFrameOrientation=orientation,
                                    meshScale=mesh_scale)
        collision_id = self._p.createCollisionShape(shapeType=p.GEOM_SPHERE,
                                    radius=0.1,\
                                    halfExtents=[0.1, 0.1, 0.1],
                                    collisionFrameOrientation=orientation,
                                    meshScale=mesh_scale)


        self.cargo_id = self._p.createMultiBody(baseMass=0.1,\
                                        baseCollisionShapeIndex=collision_id,\
                                        baseVisualShapeIndex=visual_id,\
                                        basePosition=cargo_shift)

        self._p.changeDynamics(self.cargo_id,-1, lateralFriction=self.k_friction)
        self._p.changeDynamics(self.cargo_id,-1, angularDamping=0.1)
        self._p.changeDynamics(self.cargo_id,-1, linearDamping=0.1)
        
        self.plane_id =self._p.loadURDF(self._urdfRoot + "/plane.urdf")

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
        [pybullet_envs](https://github.com/bulletphysics/bullet3/)
        used under the pybullet license available 
            -> https://github.com/bulletphysics/bullet3/blob/master/LICENSE.txt
        """

        action = action * 8.0

        # code from https://github.com/bulletphysics/bullet3/
        if (self._renders):
          basePos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
          #self._p.resetDebugVisualizerCamera(1, 30, -40, basePos)

        if (self._isDiscrete):
          fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
          steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
          forward = fwd[action]
          steer = steerings[action]
          realaction = [forward, steer]
        else:
          realaction = action

        self._racecar.applyAction(realaction)
        for i in range(self._actionRepeat):
          self._p.stepSimulation()
          #if self._renders:
          #  time.sleep(self._timeStep)
          self._observation = self.getExtendedObservation()

          if self._termination():
            break
          self._envStepCounter += 1
        reward = self._reward()
        done = self._termination()
        # end code from https://github.com/bulletphysics/bullet3/

        info = self.compute_cost()

        return np.array(self._observation), reward, done, info

    def reset(self):
        """
        the majority of this function consists of code from 
        [pybullet_envs](https://github.com/bulletphysics/bullet3/)
        used under the pybullet license available 
            -> https://github.com/bulletphysics/bullet3/blob/master/LICENSE.txt
        """
        self._timeStep = 0.01

        # code from https://github.com/bulletphysics/bullet3/

        self._p.resetSimulation()
        #p.setPhysicsEngineParameter(numSolverIterations=300)
        self._p.setTimeStep(self._timeStep)
        #self._p.loadURDF(os.path.join(os.path.dirname(__file__),"../data","plane.urdf"))
        stadiumobjects = self._p.loadSDF(os.path.join(self._urdfRoot, "stadium.sdf"))
        #move the stadium objects slightly above 0
        for i in stadiumobjects:
          pos, orn = self._p.getBasePositionAndOrientation(i)
          newpos = [pos[0], pos[1], pos[2] + 0.1]
          self._p.resetBasePositionAndOrientation(i, newpos, orn)

        dist = 5 + 2. * random.random()
        ang = 2. * 3.1415925438 * random.random()

        ballx = dist * math.sin(ang)
        bally = dist * math.cos(ang)
        ballz = 1

        self._ballUniqueId = self._p.loadURDF(os.path.join(self._urdfRoot, "sphere2red.urdf"),
                                              [ballx, bally, ballz])
        self._p.setGravity(0, 0, -10)
        self._racecar = racecar.Racecar(self._p, urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        for i in range(100):
          self._p.stepSimulation()
        self._observation = self.getExtendedObservation()
        # end code from https://github.com/bulletphysics/bullet3/ (except for the return line)

        self.make_cargo()

        return np.array(self._observation)


class CubeRacecarEnv(SphereRacecarEnv):

    def __init__(self,\
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=10,
               isEnableSelfCollision=True,
               isDiscrete=False,
               renders=True):

        self.k_friction = 0.5

        super(CubeRacecarEnv, self).__init__(urdfRoot=urdfRoot,\
                actionRepeat=actionRepeat,\
                isEnableSelfCollision=isEnableSelfCollision,\
                isDiscrete=isDiscrete,\
                renders=renders)

    def make_cargo(self):

        self.height_threshold = 0.25
        orientation = self._p.getQuaternionFromEuler([np.pi/2,0,0])
        cargo_shift = [0.15, 0.0, 0.3]
        mesh_scale = [0.1,0.1,0.1]

        visual_id = self._p.createVisualShape(shapeType=p.GEOM_BOX,
                                    halfExtents=[0.1, 0.1, 0.1],
                                    rgbaColor=[1, 0, 1, 1],
                                    specularColor=[0.8, .0, 0],
                                    visualFrameOrientation=orientation,
                                    meshScale=mesh_scale)
        collision_id = self._p.createCollisionShape(shapeType=p.GEOM_BOX,
                                    halfExtents=[0.1, 0.1, 0.1],
                                    collisionFrameOrientation=orientation,
                                    meshScale=mesh_scale)


        self.cargo_id = self._p.createMultiBody(baseMass=0.1,\
                                        baseCollisionShapeIndex=collision_id,\
                                        baseVisualShapeIndex=visual_id,\
                                        basePosition=cargo_shift)

        self._p.changeDynamics(self.cargo_id,-1, lateralFriction=self.k_friction)
        self._p.changeDynamics(self.cargo_id,-1, angularDamping=0.1)
        self._p.changeDynamics(self.cargo_id,-1, linearDamping=0.1)
        
        self.plane_id =self._p.loadURDF(self._urdfRoot + "/plane.urdf")

class DuckRacecarEnv(SphereRacecarEnv):

    def __init__(self,\
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=10,
               isEnableSelfCollision=True,
               isDiscrete=False,
               renders=True):

        self.k_friction = 0.5

        super(DuckRacecarEnv, self).__init__(urdfRoot=urdfRoot,\
                actionRepeat=actionRepeat,\
                isEnableSelfCollision=isEnableSelfCollision,\
                isDiscrete=isDiscrete,\
                renders=renders)

    def make_cargo(self):

        self.height_threshold = 0.13
        orientation = self._p.getQuaternionFromEuler([np.pi/2,0,0])
        cargo_shift = [0.15, 0.0, 0.175]
        mesh_scale = [0.1,0.1,0.1]

        visual_id = self._p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName= self._urdfRoot + "/duck.obj",
                                    radius=0.1,
                                    rgbaColor=[1, 0, 1, 1],
                                    specularColor=[0.8, .0, 0],
                                    visualFrameOrientation=orientation,
                                    meshScale=mesh_scale)
        collision_id = self._p.createCollisionShape(shapeType=p.GEOM_MESH,
                                    fileName= self._urdfRoot + "/duck.obj",
                                    radius=0.1,
                                    collisionFrameOrientation=orientation,
                                    meshScale=mesh_scale)


        self.cargo_id = self._p.createMultiBody(baseMass=0.1,\
                                        baseCollisionShapeIndex=collision_id,\
                                        baseVisualShapeIndex=visual_id,\
                                        basePosition=cargo_shift)

        self._p.changeDynamics(self.cargo_id,-1, lateralFriction=self.k_friction)
        self._p.changeDynamics(self.cargo_id,-1, angularDamping=0.1)
        self._p.changeDynamics(self.cargo_id,-1, linearDamping=0.1)
        
        self.plane_id =self._p.loadURDF(self._urdfRoot + "/plane.urdf")
if __name__ == "__main__":

    env = DuckRacecarEnv() 
    #env = gym.make("RacecarZedBulletEnv-v0")


    obs = env.reset()
    

    for step in range(30):

        env.step(np.array([1.0, -1.0]))

