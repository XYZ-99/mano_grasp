import argparse
import torch
import trimesh
import numpy as np
from manopth.manolayer import ManoLayer

import sys
import os
from os.path import join as pjoin
base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, pjoin(base_path, ".."))

from mano_grasp.kinematics import Kinematics


MANO_PATH = "/home/xuyinzhen/Documents/manopth/mano/models"

def construct_pose_and_dofs():
    def d2r(degree):
        """ degree to radian """
        return degree / 180 * np.pi
    
    pose = [0, 0, 0,
            0, 0, 0, 1]
    # dofs = [ d2r(-15), d2r(65), 0,
    #          0, d2r(85), 0,
    #          d2r(25), d2r(85), 0,
    #          d2r(10), d2r(75), 0,
    #          d2r(60), 0, 0, 0 ]
    dofs = [ 0, d2r(15), d2r(60),
             0, d2r(15), d2r(60),
             d2r(5), d2r(15), d2r(60),
             d2r(5), d2r(15), d2r(60),
             d2r(15), d2r(15), d2r(15), d2r(15) ]
    return np.array(pose), np.array(dofs)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=1000.0)
    parser.add_argument("--save-path", default="/home/xuyinzhen/Documents/mano_grasp/debug/initial_pose.obj")  # TODO
    parser.add_argument("--mano-hand-path", 
                        default="/home/xuyinzhen/Documents/mano_grasp/models/ManoHand")
    
    return parser.parse_args()

def main():
    args = parse_args()
    scale = args.scale
    save_path = args.save_path
    mano_hand_path = args.mano_hand_path
    
    """ Get mano_pose """
    k = Kinematics(mano_hand_path)
    pose, dofs = construct_pose_and_dofs()

    mano_trans, mano_pose = k.getManoPose(pose[:3], pose[3:], dofs)
    mano_pose = np.array(mano_pose).reshape(1, -1)

    """ Get mesh """
    mano_layer_right = ManoLayer(
        mano_root=MANO_PATH,
        side="right",
        use_pca=False,
        ncomps=45,
        flat_hand_mean=True
    )
    
    hand_vertices, _ = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(mano_pose), 
                                                th_trans=torch.FloatTensor(mano_trans))
    hand_vertices = hand_vertices.cpu().data.numpy()[0] / scale
    hand_faces = mano_layer_right.th_faces.cpu().data.numpy()
    saved_data = {}
    saved_data["hand_vertices"] = hand_vertices
    saved_data["hand_faces"] = hand_faces
    save_dir = os.path.dirname(save_path)
    np.savez(pjoin(save_dir, "vf_1.npz"), **saved_data)
    """ Save """
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
    with open(save_path, "w") as f:
        f.write(mesh_txt)
    
    
if __name__ == "__main__":
    main()
    