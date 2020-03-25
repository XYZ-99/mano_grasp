import yaml
import json
import itertools
import os
import numpy as np
from kinematics import Kinematics
from grasp_utils import *


class GraspSaver:
    """ Grasp saver """

    def __init__(self, path_out, dataset):
        """Constructor
        
        Arguments:
            graspit_process {GraspitProcess} -- process
        """
        self._path_out = path_out
        self._dataset = dataset

    def __call__(self, body_name, body_grasps):
        """Generated grasps for specific object
        
        Arguments:
            object_name {str} -- object
        
        Returns:
            tuple -- object_name, generated grasps
        """
        
        grasps_filename = os.path.join(self._path_out, '{}.json'.format(body_name))
        if os.path.exists(grasps_filename):
            with open(grasps_filename, 'r') as grasps_file:
                existing_grasps = json.load(grasps_file)
            body_grasps.extend(existing_grasps['grasps'])
            print('{}: loading {} grasps'.format(
                body_name,
                len(existing_grasps['grasps']),
            ))


        print('{}: saving {} grasps'.format(
            body_name,
            len(body_grasps),
        ))
        with open(grasps_filename, 'w') as grasps_file:
            scale = 1
            split = body_name.split('_scale_')
            if len(split) > 1:
                scale = float(split[1])
            object_id = split[0]
            grasps_description = {
                'grasps': body_grasps,
                'dataset': self._dataset,
                'object_scale': scale,
                'object_cat': '',
                'object_id': object_id
            }
            json.dump(grasps_description, grasps_file)
            
