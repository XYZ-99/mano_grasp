"""
python ./data_generation/record_data.py --model XXX --category bottle --grasp-num 10 --output-dir XXX
python ./data_generation/record_data.py --models-file XXX --category bottle --grasp-num 10 --output-dir XXX
"""

import argparse
import os
import os.path as osp
import numpy as np
from scipy.spatial.transform import Rotation as R

import time

import sys
sys.path.insert(0, "./mano_grasp")

from mano_grasp.graspit_process import GraspitProcess
from mano_grasp.graspit_scene import GraspitScene
from mano_grasp.grasp_miner import GraspMiner
from mano_grasp.grasp_saver import GraspSaver

def modify_plan(plan, modifying_identifier):
    """
    :param modifying_identifier: may simply be the category, or the specific obj model name
    """
    def d2r(degree):
        """ degree to radian """
        return degree / 180 * np.pi
    
    translation = plan["pose"][:3]  # 3
    xyzw = plan["pose"][3:]  # 4
    dofs = plan["dofs"]  # 13
    
    rng = np.random.default_rng()
    if modifying_identifier in ["bottle"]:
        # the unit is meter
        translation[1] = (rng.random() / 10) - 0.05  # A random shift in ±5cm
        rot0 = R.from_euler("x", -90, degrees=True)  # align with the y-axis
        
        """ compute the rotation along the y-axis """
        xz_coord = np.array([translation[0], 0, translation[2]])
        x_pos = np.array([1, 0, 0])
        cos_theta = np.dot(xz_coord, x_pos) / np.linalg.norm(xz_coord)
        if translation[2] <= 0:
            theta = np.arccos(cos_theta)
        else:
            theta = -np.arccos(cos_theta)
        rot1 = R.from_euler("y", theta, degrees=False)
        
        """ Fine tuning """
        rot2 = R.from_euler("y", -90, degrees=True)
        
        rot = rot2 * rot1 * rot0
        xyzw = rot.as_quat()
    elif modifying_identifier in ["00256_scale_1000"]:  # Too fat a bottle
        # the unit is meter
        translation[1] = (rng.random() / 10) - 0.05  # A random shift in ±5cm
        rot0 = R.from_euler("x", -90, degrees=True)  # align with the y-axis
        
        """ compute the rotation along the y-axis """
        xz_coord = np.array([translation[0], 0, translation[2]])
        x_pos = np.array([1, 0, 0])
        cos_theta = np.dot(xz_coord, x_pos) / np.linalg.norm(xz_coord)
        if translation[2] <= 0:
            theta = np.arccos(cos_theta)
        else:
            theta = -np.arccos(cos_theta)
        rot1 = R.from_euler("y", theta, degrees=False)
        
        """ Fine tuning """
        rot2 = R.from_euler("y", -75, degrees=True)
        
        rot = rot2 * rot1 * rot0
        xyzw = rot.as_quat()
    elif modifying_identifier in ["car"]:
        # the unit is meter
        translation[1] = (rng.random() * 0.05) + 0.05  # [0.05, 0.10) m height
        translation[0] /= 2
        translation[2] /= 2
        
        if translation[2] > 0.02:
            random_angle = - rng.random() * np.pi  # [-π, 0)
        elif translation[2] < -0.02:
            random_angle = rng.random() * np.pi  # [0, π)
        else:
            random_angle = rng.random() * np.pi / 3 - (np.pi / 6)  # [-π/6, π/6)
            if rng.integers(0, 2) == 1:
                random_angle += np.pi

        # random_angle = rng.random() * 2 * np.pi  # [0, 2π)
        rot0 = R.from_euler("y", random_angle, degrees=False)
        
        rot = rot0
        xyzw = rot.as_quat()
    elif modifying_identifier in ["mug"]:
        # the unit is meter
        translation[0] = (rng.random() * 0.15) + 0.05  # [0.05, 0.20)
        translation[1] = (rng.random() * 0.02) - 0.01  # [-0.01, 0.01)
        translation[2] = (rng.random() * 0.02) - 0.01  # [-0.01, 0.01)
        
        r = np.linalg.norm(translation)
        beta = np.arcsin(translation[1] / r)
        alpha = np.arcsin(translation[2] / (r * np.cos(beta)))
        
        rot0 = R.from_euler("x", -90, degrees=True)
        rot1 = R.from_euler("z", beta, degrees=False)
        rot2 = R.from_euler("y", - alpha, degrees=False)
        # y_angle = rng.random() * 30
        # rot0 = R.from_euler("y", - (75 + y_angle), degrees=True)  # [75deg, 105deg)
        # z_angle = rng.random() * 5
        # rot1 = R.from_euler("z", - (87.5 + z_angle), degrees=True)  # [87.5deg, 92.5deg)
        # x_angle = rng.random() * 10 - 5  # [-5deg, 5deg)  # This one should be imposed around the 
        
        rot = rot2 * rot1 * rot0
        xyzw = rot.as_quat()
        
        dofs = [ d2r(-5), d2r(65), 0,
             0, d2r(85), 0,
             d2r(5), d2r(85), 0,
             d2r(5), d2r(75), 0,
             d2r(60), 0, 0, 0 ]
        dofs = np.array(dofs)
        
        # if translation[0] < 0.05:
        #     translation = np.array([0.25, 0, 0])
        # else:
        #     hand_rot = R.from_quat(xyzw)
        #     hand_frame_z = hand_rot.apply([0, 0, 1])  # Should be pointing above
        #     cos_above_angle = np.dot(hand_frame_z, [0, 1, 0]) / np.linalg.norm(hand_frame_z)
        #     above_angle = np.arccos(cos_above_angle)  # < π / 2
            
        #     hand_frame_x = hand_rot.apply([1, 0, 0])  # Should be pointing outward from the mug handle
        #     cos_x_angle = np.dot(hand_frame_x, [1, 0, 0]) / np.linalg.norm(hand_frame_x)
        #     x_angle = np.arccos(cos_x_angle)  # < π / 2
        #     if above_angle >= np.pi / 2 or x_angle >= np.pi / 2:
        #         translation = np.array([0.25, 0, 0])
        
    plan["pose"][:3] = translation
    plan["pose"][3:] = xyzw
    plan["dofs"] = dofs
    return plan

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        help="Find grasps for only one model.")
    parser.add_argument("--models-file", type=str,
                        help="Find grasps for all obj models in $models_file.")
    parser.add_argument("--output-dir", type=str,
                        help="The directory to store the json results.")
    parser.add_argument("--grasp-num", type=int,
                        help="The number of grasps to be saved for each object.")
    parser.add_argument("-mim", "--modifing-id-mode", type=str, default="category",
                        help="""[category|instance]. 
                        If instance modifier is not implemented, it will fallback to category.""")
    parser.add_argument("--category", type=str, default="mug",
                        help="Specify the category if mim is category.")
    
    """ For graspit """
    parser.add_argument('-n', '--n_jobs', type=int, default=1)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-e', '--headless', action='store_true', help="Start in headless mode")
    parser.add_argument('-x',
                        '--xvfb',
                        action='store_true',
                        help="Start with Xserver Virtual Frame Buffer (Xvfb)")
    parser.add_argument('--graspit_dir',
                        type=str,
                        default=os.environ['GRASPIT'],
                        help="Path to GraspIt root directory")
    parser.add_argument('--plugin_dir',
                        type=str,
                        default=os.environ['GRASPIT_PLUGIN_DIR'],
                        help="Path to directory with a graspit_interface plugin")
    parser.add_argument('-s', '--max_steps', type=int, default=0, help="Max search steps per object")
    parser.add_argument('--relax_fingers',
                        action='store_true',
                        help="Randomize squezzed fingers positions")
    parser.add_argument('--change_speed', action='store_true', help="Try several joint's speed ratios")
    parser.add_argument('-ds', '--dataset', type=str, default='', help="dataset name")
    
    return parser.parse_args()

def ensure_dir_exists(dir):
    if not osp.exists(dir):
        os.makedirs(dir)
    
def main():
    args = parse_args()
    model = args.model
    models_file = args.models_file
    output_dir = args.output_dir
    grasp_num = args.grasp_num
    modifing_id_mode = args.modifing_id_mode
    args_category = args.category
    
    """ Object list """
    if model is not None:
        models = [model]
    elif models_file is not None:
        with open(models_file) as f:
            models = f.read().splitlines()
    else:
        raise ValueError(f"Either models or model_file should be given.")
    
    process = GraspitProcess(
        graspit_dir=args.graspit_dir,
        plugin_dir=args.plugin_dir,
        headless=args.headless,
        xvfb_run=args.xvfb,
        verbose=args.verbose
    )
    
    ensure_dir_exists(output_dir)
    saver = GraspSaver(path_out=output_dir, dataset=args.dataset)
    
    generator = GraspMiner(
        graspit_process=process,
        max_steps=args.max_steps,
        max_grasps=grasp_num,
        relax_fingers=args.relax_fingers,
        change_speed=args.change_speed,
        saver=saver,
        plan_modifier=modify_plan
    )
    
    if modifing_id_mode == "category":
        # e.g., bottle_blue_google_norm_scale_1000
        # category = models[0].split("_")[0]
        modifing_identifier = args_category
    elif modifing_id_mode == "instance":
        """
        Since this case is usually designed for a single outlier,
        we assume there is only one model in models
        """
        modifing_identifier = models[0]
    else:
        raise NotImplementedError(f"modifing_id_mode: {modifing_id_mode} not implemented!")
        
    
    if args.n_jobs > 1:
        from joblib import Parallel, delayed
        grasps = Parallel(n_jobs=args.n_jobs, verbose=50)(delayed(generator)(m,
                                                                             modifing_identifier) for m in models)
    else:
        grasps = [generator(body,
                            modifing_identifier) for body in models]
        
    if args.debug:
        with GraspitProcess(graspit_dir=args.graspit_dir, plugin_dir=args.plugin_dir) as p:
            for body_name, body_grasps in grasps:
                scene = GraspitScene(p.graspit, 'ManoHand', body_name)
                for grasp in body_grasps:
                    scene.grasp(grasp['pose'], grasp['dofs'])
                    time.sleep(5.0)
        

if __name__ == "__main__":
    main()
