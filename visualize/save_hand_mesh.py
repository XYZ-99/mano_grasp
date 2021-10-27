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
from os.path import join as pjoin
import os
base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, pjoin(base_path, ".."))

from common.data_utils import fast_load_obj
from common.vis_utils import plot_hand_w_object

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', default='bottle', help='object category for benchmarking')
    parser.add_argument('-v', '--viz', action='store_true', help='whether to visualize')
    parser.add_argument('-s', '--save', action='store_false', help='whether to save')
    args = parser.parse_args()

    # infos           = global_info()
    # my_dir          = infos.code_path
    # dset_info       = infos.datasets[args.category]
    # num_parts       = dset_info.num_parts
    # num_ins         = dset_info.num_object
    # name_dset       = dset_info.dataset_name
    # grasps_meta     = infos.grasp_path
    # mano_path       = infos.mano_path
    # hand_mesh       = infos.hand_path
    # viz_path        = infos.viz_path
    mano_path = "/home/xuyinzhen/Documents/manopth/mano/models"
    grasps_meta = "/home/xuyinzhen/Documents/mano_grasp/debug"

    mano_layer_right = ManoLayer(
            mano_root=mano_path , side='right', use_pca=False, ncomps=45, flat_hand_mean=True)

    # load glass pose in json
    # json_names = glob.glob( grasps_meta + '/*json') # eyeglasses_0002_0_scale_200.json
    json_names = glob.glob( grasps_meta + f'/{args.category}/*json')
    json_names.sort()
    print('load grasps from %s'%(pjoin(grasps_meta, args.category)))
    print('find %d grasps file.' % (len(json_names)))
    for json_name in json_names:
        name_attrs = json_name.split('.js')[0].split('/')[-1].split('_')
        if len(name_attrs) == 4: #articulated obj
            part = name_attrs[1]
        instance = name_attrs[0]
        scale    = name_attrs[-1]
        save_dir = pjoin("debug", args.category, 'scale'+scale, instance)
        viz_dir  = pjoin("debug", args.category, 'scale'+scale, instance)
        if not os.path.exists( save_dir ):
            os.makedirs(save_dir)
        if not os.path.exists( viz_dir ):
            os.makedirs(viz_dir)
        with open(json_name) as json_file:
            hand_attrs = json.load(json_file)

        if args.viz:
            objname = f'{whole_obj}/{args.category}/{instance}.obj' # TODO
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
                mesh_txt = trimesh.exchange.obj.export_obj(mesh, include_normals=False, include_color=False, include_texture=False, return_texture=False, write_texture=False, resolver=None, digits=8)
                with open(save_name,"w") as fp:
                    fp.write(mesh_txt)
                print('%s has been saved'%(save_name))

            if args.viz:
                print()
                print('The file path of this object is %s' % (objname))
                print('This is the %dth ranked grasp (%d in total)' % (j, len(hand_attrs['grasps'])))
                print('The quality of this grasp is %s' % (hand_attrs['grasps'][j]['quality']))
                plot_hand_w_object(obj_verts, obj_faces, hand_vertices, hand_faces, save_path=viz_dir + f'/{j}.png', save=False)


if __name__ == '__main__':
    main()
