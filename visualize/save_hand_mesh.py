"""
python ./visualize/save_hand_mesh.py --grasps-dir XXX
"""
import os
import argparse
import glob
import trimesh
import torch
import numpy as np
import json
from os.path import join as pjoin
from manopth.manolayer import ManoLayer

import sys
import os
base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, pjoin(base_path, ".."))

from common.data_utils import fast_load_obj
from common.vis_utils import plot_hand_w_object

MANO_PATH = "/home/xuyinzhen/Documents/manopth/mano/models"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', default='bottle', help='object category for benchmarking')
    parser.add_argument('-v', '--viz', action='store_true', help='whether to visualize')
    parser.add_argument('-s', '--save', action='store_false', help='whether to save')
    parser.add_argument('--grasps-dir', type=str, default="/home/xuyinzhen/Documents/mano_grasp/debug")
    return parser.parse_args()

def main():
    args = parse_args()
    category = args.category
    grasps_dir = args.grasps_dir

    mano_layer_right = ManoLayer(
            mano_root=MANO_PATH , side='right', use_pca=False, ncomps=45, flat_hand_mean=True)

    json_names = glob.glob(grasps_dir + f'/*.json')
    json_names.sort()
    
    print('Load grasps from %s' % (pjoin(grasps_dir, category)))
    print('Find %d grasps file.' % (len(json_names)))
    
    for json_name in json_names:
        name_attrs = json_name.split('.js')[0].split('/')[-1].split('_')
        if len(name_attrs) == 4: #articulated obj
            part = name_attrs[1]
        instance = "_".join(name_attrs[:-2])
        scale: str = name_attrs[-1]
        save_dir = pjoin("debug", category, 'scale'+scale, instance)
        viz_dir  = pjoin("debug", category, 'scale'+scale, instance)
        if not os.path.exists( save_dir ):
            os.makedirs(save_dir)
        if not os.path.exists( viz_dir ):
            os.makedirs(viz_dir)
            
        with open(json_name) as json_file:
            hand_attrs = json.load(json_file)

        if args.viz:
            objname = instance + ".obj"
            obj= fast_load_obj(open(objname, 'rb'))[0] # why it is [0]
            obj_verts = obj['vertices']
            obj_faces = obj['faces']

        for j in range(len(hand_attrs['grasps'])):
            if len(name_attrs) == 4:
                save_name = save_dir + f'/{part}_{j}.obj'
            else:
                save_name = save_dir + f'/{j}.obj'
                
            posesnew   = np.array(hand_attrs['grasps'][j]['mano_pose']).reshape(1, -1)
            mano_trans = hand_attrs['grasps'][j]['mano_trans']

            hand_vertices, _ = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(posesnew), th_trans=torch.FloatTensor(mano_trans))

            # divided by scale!!!!
            hand_vertices = hand_vertices.cpu().data.numpy()[0] / int(scale)
            hand_faces = mano_layer_right.th_faces.cpu().data.numpy()
            
            if args.save:
                mesh = trimesh.Trimesh(vertices=hand_vertices,
                           faces=hand_faces)
                mesh_txt = trimesh.exchange.obj.export_obj(mesh, 
                                                           include_normals=False, 
                                                           include_color=False, 
                                                           include_texture=False, 
                                                           return_texture=False, 
                                                           write_texture=False, 
                                                           resolver=None, 
                                                           digits=8)
                
                with open(save_name,"w") as fp:
                    fp.write(mesh_txt)

                print('%s has been saved'%(save_name))

            if args.viz:
                print()
                print('The file path of this object is %s' % (objname))
                print('This is the %dth ranked grasp (%d in total)' % (j, len(hand_attrs['grasps'])))
                print('The quality of this grasp is %s' % (hand_attrs['grasps'][j]['quality']))
                plot_hand_w_object(obj_verts, 
                                   obj_faces, 
                                   hand_vertices, 
                                   hand_faces, 
                                   save_path=viz_dir + f'/{j}.png', 
                                   save=False)


if __name__ == '__main__':
    main()
