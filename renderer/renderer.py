import trimesh
import numpy as np
import os
from multiprocessing import Process
import argparse
from tqdm import tqdm
import pyrender
import matplotlib.pyplot as plt
import cv2
from os.path import join as pjoin
import pickle
import json
from copy import deepcopy

import sys
file_base_path = os.path.dirname(__file__)
sys.path.insert(0, file_base_path)
sys.path.insert(0, pjoin(file_base_path, '..'))

from common.data_utils import rvec_from_mat, mat_from_rvec
from common.pose_inter import get_camera_random_pose, cam_pose_interp, cam_pose_to_matrix
from common.pose_util import rot_diff_degree


category2scale = {
    'car': 0.3,
    'bottle': 0.25,
    'mug': 0.25,
    'bowl': 0.3
}


def normalize(q):
    norm = np.linalg.norm(q)
    return q/norm

def rmat_to_q(rmat):
    trace = 1+rmat[0,0]+rmat[1,1]+rmat[2,2]
    if trace < 0:
        trace = 0
    r = np.sqrt(trace)
    s = 1.0 / (2 * r + 1e-7)
    w = 0.5 * r
    x = (rmat[2, 1] - rmat[1, 2]) * s
    y = (rmat[0, 2] - rmat[2, 0]) * s
    z = (rmat[1, 0] - rmat[0, 1]) * s
    q = np.array([w,x,y,z])
    return normalize(q)


def transform_pc(pts, pose):
    '''
    pts: 3 * n
    pose: 4 * 4
    '''
    points = np.concatenate([pts, np.ones_like(pts[:,0:1])], axis=1)
    points = np.matmul(pose, pts)
    points = points[:,:3]/points[:,3:]
    return points


def test_camera_interp(pose_s, pose_t, beta, rlim=60.0, tlim=0.20):
    pose = cam_pose_interp(pose_s, pose_t, beta)
    rot_diff = rot_diff_degree(pose_s['rotation'], pose['rotation'])
    pos_diff = (cam_pose_to_matrix(pose) - cam_pose_to_matrix(pose_s))[:3, 3].reshape(-1)
    pos_diff = np.linalg.norm(pos_diff)
    return rot_diff < rlim and pos_diff < tlim


def create_pose_list(category, target_length, max_segment=50):
    full_pose_list = []
    pose_s = get_camera_random_pose(category, False, to_matrix=False)

    while len(full_pose_list) < target_length:
        #sample end pose of one segment
        pose_t = get_camera_random_pose(category, False, to_matrix=False)
        #check end pose of one segment
        beta = 1.0
        while not test_camera_interp(pose_s, pose_t, beta, rlim=60, tlim=0.20):
            beta *= 0.5
        pose_t = cam_pose_interp(pose_s, pose_t, beta)

        delta = 0.5
        while not test_camera_interp(pose_s, pose_t, delta, rlim=3, tlim=0.02):
            delta *= 0.5
        cur_length = int(1.0 / delta)
        cur_length = min(cur_length, target_length - len(full_pose_list))
        cur_length = min(cur_length, max_segment)

        #create a segment
        for i in range(cur_length):
            full_pose_list.append(cam_pose_interp(pose_s, pose_t, delta*i, to_matrix=True))
            if i == cur_length - 1:
                pose_s = cam_pose_interp(pose_s, pose_t, delta*i, to_matrix=False)

    return full_pose_list


def create_partial(proc_num, base_folder, obj_path, original_obj_path, save_folder, category, ins_num, render_num, seq=False, modify=False,
                   yfov=np.deg2rad(60), pw=640, ph=480, near=0.1, far=10):

    #for blender data translation
    mm = trimesh.load(original_obj_path, force='mesh')
    blender_scale = category2scale[category]
    blender_center = np.array((mm.vertices.max(axis=0) + mm.vertices.min(axis=0)) / 2 * blender_scale)
    #load object
    print(obj_path)
    m = trimesh.load(obj_path, force='mesh')
    base_scale = np.sqrt(((m.vertices.max(axis=0)-m.vertices.min(axis=0))**2).sum())

    #pyrender load object
    scene = pyrender.Scene()
    obj_mesh = pyrender.Mesh.from_trimesh(m)
    obj_node = pyrender.Node(mesh=obj_mesh, matrix=np.eye(4))
    scene.add_node(obj_node)

    #initialize camera
    camera_pose = np.eye(4)
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=pw / ph, znear=near, zfar=far)
    projection = camera.get_projection_matrix()
    scene.add(camera, camera_pose)
    r = pyrender.OffscreenRenderer(pw, ph)

    #save folder
    if seq:
        preproc_folder = pjoin(save_folder, 'preproc', category, 'seq')
    else:
        preproc_folder = pjoin(save_folder, 'preproc', category, 'single_frame')
    os.makedirs(preproc_folder, exist_ok=True)

    #read grasp annotations in json file
    grasp_path = pjoin(base_folder, 'grasps', category, ins_num + '_scale_1000' + '.json')
    with open(grasp_path, 'r') as f:
        grasp_data = json.load(f)['grasps']

    #load hand mesh
    hand_base_path =  pjoin(base_folder, 'hands', category, 'scale1000', ins_num)
    lst = [i for i in os.listdir(hand_base_path) if '.obj' in i]
    if len(lst) == 0:
        print('find no hand in %s!'%(hand_base_path))
    blender_data = []
    for i in range(len(lst)):
        if i == 10:
            break
        hand_path = pjoin(hand_base_path, str(i)+'.obj')
        hand_m = trimesh.load(hand_path, force='mesh')
        template_T = np.array([95.6699, 6.3834, 6.1863]) / 1000  # tmd:)

        #pyrender load hand
        hand_mesh = pyrender.Mesh.from_trimesh(hand_m)
        hand_node = pyrender.Node(mesh=hand_mesh, matrix=np.eye(4))
        scene.add_node(hand_node)
        if seq:
            pose_list = create_pose_list(category, render_num)

        for j in range(render_num):
            jj = str(j).zfill(3)
            if seq:
                nocs2cam = pose_list[j]
                random_R = nocs2cam[:3,:3]
                random_T = nocs2cam[:3, 3]
            else:
                # randomly sample a pose(only T,R; NO scale!)
                #random_T = np.random.randn(3)*0.1 + mean_pose
                #random_R = trimesh.transformations.random_rotation_matrix()[:3, :3]
                #nocs2cam = np.eye(4)
                #nocs2cam[:3, 3] = random_T
                #nocs2cam[:3, :3] =  random_R
                nocs2cam = get_camera_random_pose(category, False, True)
                random_R = nocs2cam[:3, :3]
                random_T = nocs2cam[:3, 3]


            hand_pose_j = {'translation':(np.matmul(random_R, grasp_data[i]['mano_trans'][0]+template_T)-template_T+random_T),
                           'pose':deepcopy(grasp_data[i]['mano_pose'])
            }
            hand_pose_j['pose'][:3] = rvec_from_mat(np.matmul(random_R, mat_from_rvec(hand_pose_j['pose'][:3])))

            #pose inv
            cam2nocs = np.eye(4)
            cam2nocs[:3, :3] = random_R.transpose()
            cam2nocs[:3, 3] = -np.matmul(random_R.transpose(), random_T)

            #transform the object and hand
            if modify:
                print(pjoin(preproc_folder, '%s_%d_%s.npz' % (ins_num, i, jj)))  # TODO: WARNING
                data_dict = np.load(pjoin(preproc_folder, '%s_%d_%s.npz' % (ins_num, i, jj)), allow_pickle=True)['all_dict'].item()
                nocs2cam = np.eye(4)
                nocs2cam[:3,:3] = data_dict['obj_pose']['rotation']
                nocs2cam[:3, 3] = data_dict['obj_pose']['translation']

            scene.set_pose(obj_node, nocs2cam)
            scene.set_pose(hand_node, nocs2cam)

            #render to a depth image and then backproject to point clouds
            color_img, _ = r.render(scene, flags=pyrender.constants.RenderFlags.RGBA)
            seg_img, depth_buffer = r.render(scene, flags=pyrender.constants.RenderFlags.SEG,seg_node_map={obj_node:[255,0,0], hand_node:[0,255,0]})
            pts, idxs = backproject(depth_buffer, projection, near, far, from_image=False)

            mask = depth_buffer > 0
            depth_z = buffer_depth_to_ndc(depth_buffer, near, far)  # [-1, 1]
            depth_image = depth_z * 0.5 + 0.5  # [0, 1]
            depth_image = linearize_img(depth_image, near, far)  # [0, 1]
            depth_image = np.uint16((depth_image * mask) * ((1 << 16) - 1))

            #object_label=0 hand_label=1
            labels = 1*(seg_img[idxs[0], idxs[1], 1]>=255/2)

            data_dict = {'points': np.array(pts),
                         'labels': np.array(labels),
                         'obj_pose': {'translation': random_T, 'rotation': np.array(random_R), 'scale': np.array(base_scale)},
                         'hand_pose': hand_pose_j,
                         'file_name': '%s_%d_%s'%(ins_num,i,jj),
                         'grasp_json_path': grasp_path,
                         'category': category,
                         'ins_num': ins_num,
                         'hand_num':i,
                         'sample_num':j,
            }

            blender_data.append({
                'hand_mesh_path': pjoin('hands', category, 'scale1000', ins_num, str(i)+'.obj'),
                'obj_mesh_path': pjoin('nocs_obj_models', original_obj_path.split('nocs_obj_models/')[-1]),
                'hand_pose': {'trans': random_T, 'q': rmat_to_q(random_R)},
                'obj_pose': {'trans': random_T - np.matmul(random_R, blender_center), 'q': rmat_to_q(random_R),'scale': blender_scale},
                'file_name': '%s_%d_%s'%(ins_num,i,jj)
            })
            # save depth image, mask image and point clouds
            if seq:
                ins_folder = pjoin(save_folder, 'img', category, 'seq', '%s_%d_%s' % (ins_num, i, jj))
                video_folder = pjoin(save_folder, 'img', category, 'video', '%s_%d' % (ins_num, i))
                os.makedirs(video_folder, exist_ok=True)
                cv2.imwrite(pjoin(video_folder, 'rgb_%s.png' % (jj)), color_img)#fake rgb
            else:
                ins_folder = pjoin(save_folder, 'img', category, 'single_frame', '%s_%d_%s' % (ins_num, i, jj))

            os.makedirs(ins_folder, exist_ok=True)
            cv2.imwrite(pjoin(ins_folder, 'depth.png'), depth_image)
            cv2.imwrite(pjoin(ins_folder, 'mask.png'), seg_img)
            #cv2.imwrite(pjoin(ins_folder, 'rgb.png'), color_img) #fake rgb

            np.savez_compressed(pjoin(preproc_folder, '%s_%d_%s.npz' % (ins_num, i, jj)), all_dict=data_dict)

            #transform the object and hand back
            scene.set_pose(obj_node, cam2nocs)
            scene.set_pose(hand_node, cam2nocs)

        scene.remove_node(hand_node)

        #each instance one video
        if seq:
            break

    return blender_data, projection, near, far


def proc_render(proc_num, path_list, base_folder, save_folder, mapping, category, render_num, seq, modify,
                yfov, pw, ph, near, far):
    blender_lst = []
    for read_path in tqdm(path_list):
        ins_num = read_path.split('/')[-1][:5]
        original_obj_abs_path = pjoin(mapping[int(ins_num)].replace('\n', ''), 'model.obj')
        blender_data, projection, near, far = create_partial(proc_num, base_folder, read_path, original_obj_abs_path, save_folder, category,
                                               ins_num, render_num, seq, modify,
                                               yfov, pw, ph, near, far)
        blender_lst.extend(blender_data)
    mode = 'single_frame' if not seq else 'seq'
    blender_anno_folder = pjoin(base_folder, 'blender_anno', category, mode)
    os.makedirs(blender_anno_folder, exist_ok=True)
    np.save(pjoin(blender_anno_folder, f'blender_anno_{proc_num}'), blender_lst)

    if proc_num == 0:
        meta_path = pjoin(save_folder, 'meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump({'near': near, 'far': far, 'projection': projection}, f)

def ndc_depth_to_buffer(z, near, far):  # z in [-1, 1]
    return 2 * near * far / (near + far - z * (far - near))


def buffer_depth_to_ndc(d, near, far):  # d in (0, +
    return ((near + far) - 2 * near * far / np.clip(d, a_min=1e-6, a_max=1e6)) / (far - near)


def linearize_img(d, near, far):  # for visualization only
    return 2 * near / (near + far - d * (far - near))


def inv_linearize_img(d, near, far):  # for visualziation only
    return (near + far - 2 * near / d) / (far - near)


def backproject(depth, projection, near, far, from_image=False, vis=False):
    proj_inv = np.linalg.inv(projection)
    height, width = depth.shape
    non_zero_mask = (depth > 0)
    idxs = np.where(non_zero_mask)
    depth_selected = depth[idxs[0], idxs[1]].astype(np.float32).reshape((1, -1))
    if from_image:
        z = depth_selected / ((1 << 16) - 1)  # [0, 1]
        z = inv_linearize_img(z, near, far)  # [0, 1]
        z = z * 2 - 1.0  # [-1, 1]
        d = ndc_depth_to_buffer(z, near, far)
    else:
        d = depth_selected
        z = buffer_depth_to_ndc(d, near, far)

    grid = np.array([idxs[1] / width * 2 - 1, 1 - idxs[0] / height * 2])  # ndc [-1, 1]

    ones = np.ones_like(z)
    pts = np.concatenate((grid, z, ones), axis=0) * d  # before dividing by w, w = -z_world = d

    pts = proj_inv @ pts
    pts = np.transpose(pts)

    pts = pts[:, :3]

    if vis:
        pmin, pmax = pts.min(axis=0), pts.max(axis=0)
        center = (pmin + pmax) * 0.5
        lim = max(pmax - pmin) * 0.5 + 0.2

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(depth, cmap=plt.cm.gray_r)
        ax = plt.subplot(1, 2, 2, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.8, s=1)
        ax.set_xlim3d([center[0] - lim, center[0] + lim])
        ax.set_ylim3d([center[1] - lim, center[1] + lim])
        ax.set_zlim3d([center[2] - lim, center[2] + lim])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    return pts, idxs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rotate_num', type=int, default=10)
    parser.add_argument('-c','--category', type=str, default='bottle')
    parser.add_argument('-m', '--modify', action='store_true', help='If true, load old annotations. Used to update or debug old data.')
    parser.add_argument('--seq', action='store_true', help='If true, randomly sample two points and generate a video by interpolation')
    parser.add_argument('-n', '--num_workers', type=int, default=4)
    parser.add_argument("--data-path", type=str, default="/home/xuyinzhen/Documents/mano_grasp")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    obj_folder = pjoin(data_path, 'models', args.category)  # obj models of objects
    save_folder = pjoin(data_path, 'render')
    os.makedirs(save_folder, exist_ok=True)


    obj_path_list = [pjoin(obj_folder, i) for i in os.listdir(obj_folder) if i.endswith('obj')]
    with open(pjoin(obj_folder, 'nocs_mapping.txt'), 'r') as f:
        mapping = f.readlines()

    yfov = np.deg2rad(60)
    pw = 512
    ph = 424
    near = 0.1
    far = 10

    per_worker_obj_list = []
    length = len(obj_path_list) // args.num_workers
    start = 0
    end = length
    for i in range(args.num_workers):
        if i == args.num_workers - 1:
            per_worker_obj_list.append(obj_path_list[start:])
        else:
            per_worker_obj_list.append(obj_path_list[start: end])
        start += length
        end += length

    process_lst = []
    for proc_num in range(args.num_workers):
        p = Process(target=proc_render, args=(proc_num, per_worker_obj_list[proc_num], data_path, save_folder, mapping,
                                              args.category, args.rotate_num, args.seq, args.modify,
                                              yfov, pw, ph, near, far))
        p.start()
        process_lst.append(p)

    for p in process_lst:
        p.join()
        