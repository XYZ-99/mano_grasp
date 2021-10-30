import json
import itertools
import os
import numpy as np
from graspit_process import GraspitProcess
from graspit_scene import GraspitScene
from kinematics import Kinematics
from grasp_utils import *


class GraspMiner:
    """ Grasp generator """

    def __init__(self,
                 graspit_process,
                 max_steps=0,
                 max_grasps=0,
                 relax_fingers=False,
                 change_speed=False,
                 robot_names=['ManoHand'],
                 saver=None,
                 plan_modifier=None):
        """Constructor
        
        Arguments:
            graspit_process {GraspitProcess} -- process
        
        Keyword Arguments:
            max_steps {int} -- max search steps per object (default: {auto})
            max_grasps {int} -- return only N best grasps per object (default: {auto})
            change_speed {bool} -- try several joint's speed ratios (default: {False})
            relax_fingers {bool} -- randomize angles of squezzed fingers (default: {False})
        """
        self._process = graspit_process
        self._max_steps = max_steps
        self._max_grasps = max_grasps
        self._relax_fingers = relax_fingers
        self._robot_names = robot_names
        self._saver = saver
        self._plan_modifier = plan_modifier
        # we can't change a joints speed ratios on the fly, so use several hand models
        if change_speed:
            self._robot_names += ['ManoHand_v2', 'ManoHand_v3']

    def __call__(self, object_name, modifying_identifier=None):
        """Generated grasps for specific object
        
        Arguments:
            object_name {str} -- object
        
        Returns:
            tuple -- object_name, generated grasps
        """
        if not self._process.run:
            self._process.start()

        grasps_all = []
        collision_path = os.path.join(self._process.dir,
            'models/objects/{}_collision.json'.format(object_name))
        collisions = []
        if os.path.exists(collision_path):
            collisions = json.load(open(collision_path, 'r'))
        if not collisions:
            collisions = [None]
        for robot_name, collision in zip(itertools.cycle(self._robot_names), collisions):
            # load hand and body and optionally collision object
            gripper = None
            if (collision is not None) and 'gripper' in collision.keys():
                gripper = collision['gripper']
            scene = GraspitScene(self._process.graspit, robot_name, object_name, gripper)

            # plan grasps with a standart procedure
            plans = scene.planGrasps(max_steps=self._max_steps)

            # execute grasps with different euristics
            variants = (
                dict(approach=False, auto_open=False),  #
                dict(approach=False, auto_open=True, full_open=True),
                dict(approach=True, auto_open=True, full_open=False),
                dict(approach=True, auto_open=True, full_open=True))

            grasps = []
            for plan in plans:
                if self._plan_modifier is not None \
                and modifying_identifier is not None:
                    plan = self._plan_modifier(plan, modifying_identifier)
                    
                pose = plan['pose']
                dofs = plan['dofs']
                # with open("/home/xuyinzhen/Documents/mano_grasp/debug/debug.txt", "w+") as f:
                #     f.write(f"pose: {pose}\n")
                #     f.write(f"dofs: {dofs}\n")
                #     f.write(f"len(pose): {len(pose)}\n")
                #     f.write(f"len(dofs): {len(dofs)}\n")

                for args in variants:
                    grasp = scene.grasp(pose, dofs, object_name, **args)
                    if grasp is not None:
                        grasps.append(grasp)

            # sort by quality
            grasps.sort(key=lambda g: g['quality'], reverse=True)

            # cut best grasps
            if self._max_grasps > 0:
                grasps = grasps[:self._max_grasps]

            # GraspIt has a tendency to squeeze the fingers even
            # they aren't in contact with an object.
            # Below we randomize joints positions in such case
            if self._relax_fingers:
                rs = np.random.RandomState(0)
                for i, joints in squeezed(grasps):
                    pose = grasps[i]['pose']
                    dofs = list(grasps[i]['dofs'])
                    for attempt in range(20):
                        angles = rs.uniform(0.0, 2.0, size=len(joints))
                        for j, a in zip(joints, angles):
                            dofs[j] = a
                        grasp = scene.grasp(pose, dofs, object_name)
                        if grasp is not None:
                            grasps[i]['dofs'] = tuple(dofs)

            grasps_all.extend(grasps)

        if self._saver:
            self._saver(object_name, grasps_all)
        return (object_name, grasps_all)
