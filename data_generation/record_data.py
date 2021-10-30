"""
python ./data_generation/record_data.py --model-path XXX --grasps-num 10 --output-dir XXX
python ./data_generation/record_data.py --models-dir XXX --grasps-num 10 --output-dir XXX
"""

import argparse
import os
import os.path as osp
import glob
from os.path import join as pjoin
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
    translation = plan["pose"][:3]  # 3
    xyzw = plan["pose"][3:]  # 4
    if modifying_identifier in ["bottle"]:
        translation[1] = 0
        xyzw = np.array([0, 0, 0, 1])
    
    plan["pose"][:3] = translation
    plan["pose"][3:] = xyzw
    return plan

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        help="Find grasps for only one model.")
    parser.add_argument("--models-dir", type=str,
                        help="Find grasps for all obj models under $models_dir.")
    parser.add_argument("--output-dir", type=str,
                        help="The directory to store the json results.")
    parser.add_argument("--grasp-num", type=int,
                        help="The number of grasps to be saved for each object.")
    parser.add_argument("-mim", "--modifing-id-mode", type=str, default="category",
                        help="""[category|instance]. 
                        If instance modifier is not implemented, it will fallback to category.""")
    
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
    model_path = args.model_path
    models_dir = args.models_dir
    output_dir = args.output_dir
    grasp_num = args.grasp_num
    modifing_id_mode = args.modifing_id_mode
    
    """ Object list """
    if model_path is not None:
        model_paths = [model_path]
    elif models_dir is not None:
        model_filewholenames = glob.glob(pjoin(models_dir, "*.obj"))
        model_paths = list(
            map(
                lambda file: pjoin(models_dir, file),
                model_filewholenames
            )
        )
    else:
        raise ValueError(f"Either models_dir or model_path should be given.")
    
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
        # e.g., /home/xuyinzhen/Documents/mano_grasp/models/Bottle/bottle_blue_google_norm.obj
        category = osp.basename(model_paths[0]).split("_")[0]
        modifing_identifier = category
    elif modifing_id_mode == "instance":
        """
        Since this case is usually designed for a single outlier,
        we assume there is only one model in model_paths
        """
        modifing_identifier = osp.basename(model_paths[0])
    else:
        raise NotImplementedError(f"modifing_id_mode: {modifing_id_mode} not implemented!")
        
    
    if args.n_jobs > 1:
        from joblib import Parallel, delayed
        grasps = Parallel(n_jobs=args.n_jobs, verbose=50)(delayed(generator)(m,
                                                                             modifing_identifier) for m in model_paths)
    else:
        grasps = [generator(body,
                            modifing_identifier) for body in model_paths]
        
    if args.debug:
        with GraspitProcess(graspit_dir=args.graspit_dir, plugin_dir=args.plugin_dir) as p:
            for body_name, body_grasps in grasps:
                scene = GraspitScene(p.graspit, 'ManoHand', body_name)
                for grasp in body_grasps:
                    scene.grasp(grasp['pose'], grasp['dofs'])
                    time.sleep(5.0)
        

if __name__ == "__main__":
    main()
