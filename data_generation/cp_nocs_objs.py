import os
from os.path import join as pjoin
import argparse
import trimesh
import transforms3d as tf
import numpy as np
from tqdm import tqdm


category2scale = {
    'car': 0.3,
    'bottle': 0.25,
    'mug': 0.25,
    'bowl': 0.3
}

category2id = {
    'car': '02958343',
    'bottle': '02876657',
    'mug': '03797390',
    'bowl': '02880940'
}

category2axis = {
    'bottle': [0, 0, 0],
    'car': [1, 0, 0],
    'mug': [0, 0, 0],
    'bowl': [0, 0, 1]
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='copy NOCS object for prepare_objects.py')
    parser.add_argument('-s', '--source_folder', 
                        type=str, 
                        default='/home/xuyinzhen/Documents/obj_models', 
                        help="NOCS source folder relative to data_path in global_info")
    parser.add_argument('-c', '--category', 
                        type=str, 
                        default='car', 
                        help="NOCS category id")
    parser.add_argument('-t', '--target_folder', 
                        type=str, 
                        default='/home/xuyinzhen/Documents/mano_grasp/models', 
                        help="target folder relative to data_path in global_info")
    parser.add_argument('-o', '--file_out', type=str, default='', help="list of prepared object versions")
    parser.add_argument('-a', '--angle', type=int, default=0, help='rotate args.angle*90 degrees. to generate better grasp pose in graspit')
    args = parser.parse_args()

    s_folder = args.source_folder
    target_folder = pjoin(args.target_folder, args.category)

    print('Copy from %s to %s' % (s_folder, target_folder))
    os.makedirs(target_folder, exist_ok=True)

    mapping_list = []
    for mode in ['train', 'val']:
        source_folder = pjoin(s_folder, mode, category2id[args.category])
        tmp = os.listdir(source_folder)
        mapping_list.extend([pjoin(source_folder, i) for i in tmp])

    save_str = '\n'.join(mapping_list)
    with open(pjoin(target_folder, 'nocs_mapping.txt'), 'w') as f:
        f.write(save_str)

    angle = [0, 0, 0]
    for i in range(3):
        angle[i] = category2axis[args.category][i] * args.angle * np.pi / 2
    r = tf.euler.euler2mat(angle[0], angle[1], angle[2], 'rxyz')
    print(r)

    count = 0
    for path in tqdm(mapping_list):
        source = pjoin(path, 'model.obj')
        target = pjoin(target_folder, str(count).rjust(5,'0') + '.obj')
        mesh = trimesh.load(source, force='mesh')
        c = (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) / 2

        mesh.vertices = np.matmul(mesh.vertices - c, r.transpose(-1,-2)) * category2scale[args.category]
        mesh_txt = trimesh.exchange.obj.export_obj(mesh, include_normals=False, include_color=False, include_texture=False,
                                                   return_texture=False, write_texture=False, resolver=None, digits=8)
        
        with open(target, "w") as fp:
            fp.write(mesh_txt)

        count += 1
