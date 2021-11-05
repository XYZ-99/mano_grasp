import sapien.core as sapien
import trimesh
import argparse
from os.path import join as pjoin
import os
import numpy as np
from PIL import Image
import json
import transforms3d as tf
from copy import deepcopy

import sys
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, pjoin(BASEPATH, ".."))

from common.pose_inter import sapien_get_camera_random_pose, sapien_cam_pose_interp, sapien_cam_pose_to_matrix, lerp
from common.pose_util import rot_diff_degree
from common.detect_collision_in_graspit import check_collision
from common.data_utils import mat_from_rvec
import cv2
from multiprocessing import Process

class global_info(object):
    def __init__(self):
        super().__init__()
        self.h2o_data_path = "/home/xuyinzhen/Documents/mano_grasp/h2o_data"
        self.partnet_path = pjoin(self.h2o_data_path, "partnet")
        self.render_path = pjoin(self.h2o_data_path, "render")
        self.hand_path = pjoin(self.h2o_data_path, "hands")
        self.grasp_path = pjoin(self.h2o_data_path, "grasps")
        self.obj_path = pjoin(self.h2o_data_path, "objs")
        self.blender_anno_path = pjoin(self.h2o_data_path, "blender_anno")


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

def rvec_from_mat(mat):
    axis, angle = tf.axangles.mat2axangle(mat, unit_thresh=1e-03)
    rvec = axis * angle
    return rvec


class BaseEnv:
    def __init__(self, save_path, sample_num, category='laptop', seq=False):
        engine = sapien.Engine()
        renderer = sapien.VulkanRenderer()
        engine.set_renderer(renderer)

        scene = engine.create_scene()
        scene.set_timestep(1 / 100.0)

        rscene = scene.get_renderer_scene()
        rscene.set_ambient_light([0.5, 0.5, 0.5])
        rscene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        rscene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        rscene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        rscene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

        # camera
        near, far = 0.1, 10
        width, height = 512, 424
        camera_mount_actor = scene.create_actor_builder().build_kinematic()
        camera = scene.add_mounted_camera(
            name="camera",
            actor=camera_mount_actor,
            pose=sapien.Pose(),  # relative to the mounted actor
            width=width,
            height=height,
            fovx=np.deg2rad(70.6),
            fovy=np.deg2rad(60),
            near=near,
            far=far,
        )

        if category == 'laptop':
            self.total_part = 2
            self.joint_type = 'revolute'
        self.category = category
        self.sample_num = sample_num

        self.engine = engine
        self.rscene = rscene
        self.scene = scene
        self.camera = camera
        self.camera_mount_actor = camera_mount_actor
        self.obj_mesh_folder = pjoin(save_path, '..', 'objs', category)
        self.seq = seq
        mode = 'seq' if seq else 'single_frame'
        self.img_folder = pjoin(save_path, 'img', category, mode)
        self.preproc_folder = pjoin(save_path, 'preproc', category, mode)
        os.makedirs(self.preproc_folder, exist_ok=True)
        os.makedirs(self.img_folder, exist_ok=True)

    def add_urdf(self, urdf_path):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        model = loader.load(urdf_path)
        assert model is not None, f'{urdf_path} not found'
        return model

    def set_object(self, urdf_path):
        self.model = self.add_urdf(urdf_path)

        # set seg dict
        seg_dict = {}
        for link in self.model.get_links():
            link_name = link.get_name()
            if link_name.startswith('link_'):  # not base
                seg_id = int(link_name.split('_')[-1])
                seg_dict[link.get_id()] = seg_id
        max_key = np.max(list(seg_dict.keys())) + 1
        num_parts = np.max(list(seg_dict.values())) + 1
        seg_array = np.ones((max_key), dtype=int) * num_parts  # map the void points to "ground"
        for key, value in seg_dict.items():
            seg_array[key] = value
        self.gt_seg_dict = seg_array


        #set joint limits
        active_joints = self.model.get_active_joints()
        qlimits = [joint.get_limits() for joint in active_joints if joint is not None]
        qlimits_l = np.concatenate([limits[..., 0] for limits in qlimits])
        qlimits_r = np.concatenate([limits[..., 1] for limits in qlimits])
        self.model_dof = len(qlimits)
        self.model_limits = (qlimits_l, qlimits_r)

    def get_object_random_pose(self, stress_joint=False, up=None):
        delta = np.random.rand(*self.model_limits[0].shape)
        if stress_joint:
            if up is None:
                up = np.random.randint(0, 2, delta.shape)
            delta = delta * 0.2
            delta = delta * (1 - up) + (1 - delta) * up
        return delta * self.model_limits[1] + (1 - delta) * self.model_limits[0], up

    #def get_object_random_qpose(self):
    #    delta = np.random.rand(*self.model_limits[0].shape)
    #    qpos = delta * self.model_limits[1] + (1 - delta) * self.model_limits[0]
    #    return qpos

    def get_random_sequence(self, target_length, max_segment=50, limited=None, stress_joint=True):
        pose_s = sapien_get_camera_random_pose(self.category, limited, to_matrix=False)
        qpos_s, up_s = self.get_object_random_pose(stress_joint=stress_joint)

        def full_interp(pose_s, pose_t, qpos_s, qpos_t, alpha, to_matrix=False):
            pose = sapien_cam_pose_interp(pose_s, pose_t, alpha, to_matrix=to_matrix)
            qpos = lerp(qpos_s, qpos_t, alpha)
            return (pose, qpos)

        def test_camera_interp(pose_s, pose_t, beta, rlim=60.0, tlim=0.20):
            pose = sapien_cam_pose_interp(pose_s, pose_t, beta)
            rot_diff = rot_diff_degree(pose_s['rotation'], pose['rotation'])
            pos_diff = (sapien_cam_pose_to_matrix(pose) - sapien_cam_pose_to_matrix(pose_s))[:3, 3].reshape(-1)
            pos_diff = np.linalg.norm(pos_diff)
            return rot_diff < rlim and pos_diff < tlim

        def test_interp(pose_s, pose_t, qpos_s, qpos_t, alpha, joint_type='revolute', rlim=3.0, tlim=0.02):
            pose, qpos = full_interp(pose_s, pose_t, qpos_s, qpos_t, alpha, to_matrix=False)
            rot_diff = rot_diff_degree(pose_s['rotation'], pose['rotation'])
            pos_diff = (sapien_cam_pose_to_matrix(pose) - sapien_cam_pose_to_matrix(pose_s))[:3, 3].reshape(-1)
            pos_diff = np.linalg.norm(pos_diff)
            if joint_type == 'revolute':
                rot_diff += np.rad2deg(np.max(np.abs(qpos - qpos_s)))
            else:
                pos_diff += np.max(np.abs(qpos - qpos_s))
            # print(f'alpha = {alpha}, rot_diff = {rot_diff}, trans_diff = {pos_diff}')
            if alpha < 1e-8:
                print('original diff')
                rot_diff = rot_diff_degree(pose_s['rotation'], pose_t['rotation'])
                pos_diff = (sapien_cam_pose_to_matrix(pose_s) - sapien_cam_pose_to_matrix(pose_t))[:3, 3].reshape(-1)
                print('rot', rot_diff, 'pos', pos_diff)
                sys.exit(0)
            return rot_diff < rlim and pos_diff < tlim

        full_pose_list = []

        while len(full_pose_list) < target_length:
            pose_t = sapien_get_camera_random_pose(self.category, limited, to_matrix=False)
            if stress_joint:
                beta = 1.0
                while not test_camera_interp(pose_s, pose_t, beta):
                    beta *= 0.5
                pose_t = sapien_cam_pose_interp(pose_s, pose_t, beta)

            qpos_t, up_t = self.get_object_random_pose(stress_joint=stress_joint, up=(1 - up_s))

            alpha = 0.5
            while not test_interp(pose_s, pose_t, qpos_s, qpos_t, alpha, joint_type=self.joint_type):
                alpha *= 0.5
            cur_length = int(1.0 / alpha)
            cur_length = min(cur_length, target_length - len(full_pose_list))
            cur_length = min(cur_length, max_segment)

            for i in range(cur_length):
                full_pose_list.append(full_interp(pose_s, pose_t, qpos_s, qpos_t, alpha * i, to_matrix=True))
                if i == cur_length - 1:
                    pose_s, qpos_s = full_interp(pose_s, pose_t, qpos_s, qpos_t, alpha * i, to_matrix=False)
                    up_s = up_t

        return full_pose_list

    def sample_and_render(self, urdf_path, grasp_path, check_obj_path_list, check_new_name_list, hand_part, instance, hand_num):
        self.set_object(urdf_path)
        obj_center_list = []
        obj_scale_list = []
        for i in range(self.total_part):
            obj_mesh_path = pjoin(self.obj_mesh_folder, f'{instance}_{i}.obj')
            print(obj_mesh_path)
            obj_mesh = trimesh.load(obj_mesh_path, force='mesh')
            obj_center_list.append((obj_mesh.vertices.max(axis=0)+obj_mesh.vertices.min(axis=0))/2)
            obj_scale_list.append(np.linalg.norm(obj_mesh.vertices.max(axis=0)-obj_mesh.vertices.min(axis=0), axis=-1))
        # read hand mano pose
        with open(grasp_path, 'r') as f:
            grasp_data = json.load(f)['grasps']

        if self.seq:
            pose_list = self.get_random_sequence(self.sample_num)

        success_sample_num = 0
        for i in range(self.sample_num):
            ii = str(success_sample_num).zfill(3)
            #get joint position
            joints = self.model.get_active_joints()
            parent_name = joints[0].get_parent_link().get_name()
            child_name = joints[0].get_child_link().get_name()
            joint_trans = joints[0].get_pose_in_parent().to_transformation_matrix()[:3, 3]
            #sample and set pose
            if self.seq:
                pos, qpos = pose_list[i]
            else:
                qpos,_ = self.get_object_random_pose()
                pos = sapien_get_camera_random_pose(self.category, False, True)

            self.model.set_qpos(qpos)
            self.camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(pos))

            #get gt pose
            gt_obj_pose_list = []
            links = self.model.get_links()
            #must in order!!!!!!!
            for num in range(total_part+1):
                for link in links:
                    name = link.get_name()
                    if name == 'link_' + str(num):
                        j_mat = np.eye(4)
                        if name == child_name or (int(child_name[-1]) == hand_part and name == f'link_{self.total_part}'):
                            j_mat[:3, 3] = -joint_trans
                        gt_obj_pose_list.append(np.matmul(link.get_pose().to_transformation_matrix(), j_mat))

            #check collision
            #check_pose_list = [gt_obj_pose_list[1-hand_part],gt_obj_pose_list[hand_part], gt_obj_pose_list[2]]
            check_pose_list = [gt_obj_pose_list[1 - hand_part],  gt_obj_pose_list[2]]

            if not check_collision(check_obj_path_list, check_pose_list, check_new_name_list):
                print('collision!')
                if self.seq:
                    break
                else:
                    continue
            else:
                print(f'{urdf_path} is ok!')

            self.scene.step()
            self.scene.update_render()
            self.camera.take_picture()

            # segmentation
            full_segmentation = self.camera.get_actor_segmentation()
            seg_img = self.gt_seg_dict[full_segmentation]
            seg = seg_img.astype(np.int8).reshape(-1)
            # reject heavy occlusion
            reject_flag = False
            for part in range(self.total_part+1):
                if len(np.where(seg==part)[0]) < 20:
                    reject_flag = True
                    break
            if reject_flag and not self.seq:
                continue
            img_folder = pjoin(self.img_folder, f'{instance}_{hand_part*20+hand_num}')
            os.makedirs(img_folder, exist_ok=True)

            #rgba
            rgba = self.camera.get_float_texture('Color')
            rgba = (rgba * 255).clip(0, 255).astype("uint8")
            rgba = Image.fromarray(rgba)
            rgb_path = pjoin(img_folder, f'{ii}_rgb.png')
            rgba.save(rgb_path)
            seg_path = pjoin(img_folder, f'{ii}_seg.png')
            cv2.imwrite(seg_path, seg_img)
            print(f'save rgb img to {rgb_path}')

            #point cloud
            position = self.camera.get_float_texture('Position')
            points_opengl = position[..., :3][position[..., 3] > 0]

            # remove bg pc
            seg_fg = np.where(seg != total_part + 1)[0]
            seg = seg[seg_fg]
            points_opengl = points_opengl[seg_fg, :]

            cam_pos = self.camera.get_model_matrix() #TODO: WHY cam_pos!=pos????
            #points_opengl = points_opengl @ cam_pos[:3, :3].T + cam_pos[:3, 3] #cam to world

            #consider camera pose. Both obj and hand
            for j in range(len(gt_obj_pose_list)):
                gt_obj_pose_list[j] = np.matmul(np.linalg.inv(cam_pos), gt_obj_pose_list[j])

            # obj gt
            obj_pose = []
            for j in range(self.total_part):
                obj_pose.append({
                    'translation': gt_obj_pose_list[j][:3,3] + np.matmul(gt_obj_pose_list[j][:3, :3], obj_center_list[j]),
                    'rotation': gt_obj_pose_list[j][:3, :3],
                    'scale': obj_scale_list[j],
                })
            '''
            #visualize
            seg_idx = np.where(seg==0)[0]
            seg_idx2 = np.where(seg == 1)[0]
            seg_idx3 = np.where(seg == 2)[0]
            seg_idx = np.random.permutation(seg_idx)
            seg_idx2 = np.random.permutation(seg_idx2)
            seg_idx3 = np.random.permutation(seg_idx3)
            p1 = points_opengl[seg_idx][:1024,:]
            p2 = points_opengl[seg_idx2][:1024,:]
            p3 = points_opengl[seg_idx3][:1024,:]
            def transform(p, i):
                pp = np.matmul(p - obj_pose[i]['translation'], obj_pose[i]['rotation']) / obj_pose[i]['scale']
                aa = np.stack((pp.max(axis=0), pp.min(axis=0)),axis=0)
                return [pp, aa]
            plot3d_pts([[p1,p2,p3],transform(p1,0), transform(p2,1)], show_fig=True, save_fig=False)
            '''
            #hand gt
            template_T = np.array([95.6699, 6.3834, 6.1863]) / 1000
            hand_R = gt_obj_pose_list[-1][:3, :3]
            hand_T = gt_obj_pose_list[-1][:3, 3]
            hand_pose_j = {'translation': (np.matmul(hand_R, grasp_data[hand_num]['mano_trans'][0] + template_T) - template_T + hand_T),
                           'pose': deepcopy(grasp_data[hand_num]['mano_pose'])
                           }
            hand_pose_j['pose'][:3] = rvec_from_mat(np.matmul(hand_R, mat_from_rvec(hand_pose_j['pose'][:3])))

            data_dict = {'points': np.array(points_opengl),
                         'labels': np.array(seg),
                         'obj_pose': obj_pose,
                         'hand_pose': hand_pose_j,
                         'file_name': '%s_%d_%s' % (instance, hand_part*20+hand_num, ii),
                         'grasp_json_path': grasp_path,
                         'category': self.category,
                         'ins_num': instance,
                         'hand_num': hand_part*20+hand_num,
                         'sample_num': ii,
                         'urdf_path': urdf_path,
                         'for_kuafu': {'pos': pos, 'qpos': qpos}
                         }

            np.savez_compressed(pjoin(self.preproc_folder, '%s_%d_%s.npz' % (instance, hand_part*20+hand_num, ii)), all_dict=data_dict)
            success_sample_num+=1
            #each grasp only take 10!
            if success_sample_num == 10:
                break
        return True


def create_obj_hand_urdf(urdf_path, hand_path, save_path, total_part):
    with open(urdf_path, 'r') as f:
        lines = f.readlines()
    part_num = os.path.basename(hand_path).split('_')[0]
    for i in range(len(lines)):
        if f'name="link_{part_num}' in lines[i]:
            origin = '<origin xyz="0 0 0"/>\n'
            for j in range(i+1, len(lines)):
                if '</link>' in lines[j]:
                    break
                if '<origin' in lines[j]:
                    origin = lines[j]
                    break
            #add a new link
            lines.insert(i, f'<link name="link_{total_part}">\n')
            # write <visual>
            lines.insert(i + 1, '<visual name="hand">\n')
            lines.insert(i + 2, origin)
            lines.insert(i + 3, '<geometry>\n')
            lines.insert(i + 4, f'<mesh filename="{hand_path}"/>\n')
            lines.insert(i + 5, '</geometry>\n')
            lines.insert(i + 6, '</visual>\n')
            lines.insert(i + 7, '</link>\n')
            #add a new joint
            lines.insert(i + 8, f'<joint name="joint_{total_part}" type="fixed">\n')
            lines.insert(i + 9, f'<child link="link_{total_part}"/>\n')
            lines.insert(i + 9, f'<parent link="link_{part_num}"/>\n')
            lines.insert(i + 11, '</joint>\n')
            break

    with open(save_path, 'w') as f:
        f.write(''.join(lines))
    return True

def proc_work(instance_lst, proc_num, infos, args):
    obj_folder = pjoin(infos.partnet_path, args.category)  # .urdf
    hand_folder = pjoin(infos.hand_path, args.category, f'scale{args.scale}')  # .obj
    for instance in instance_lst:
        urdf_path = pjoin(obj_folder, instance, 'mobility.urdf')
        hand_lst = os.listdir(pjoin(hand_folder, instance))

        for hand_file in hand_lst:
            if '.obj' not in hand_file:
                continue
            hand_path = pjoin(hand_folder, instance, hand_file)
            new_urdf_save_path = pjoin(obj_folder, instance, hand_file[:-4]+'.urdf')
            print(new_urdf_save_path)
            create_obj_hand_urdf(urdf_path, hand_path, new_urdf_save_path, total_part)
            
            #which part hand touch
            hand_part = int(hand_file[0])

            #for hand annotation
            grasp_path = pjoin(infos.grasp_path, args.category, f'{instance}_{hand_part}_scale_{args.scale}.json')

            #for collision detection
            check_obj_path_list = []
            check_obj_path_list.append(pjoin(infos.obj_path, args.category, f'{instance}_{1-hand_part}.obj'))
          #  check_obj_path_list.append(pjoin(infos.obj_path, args.category, f'{instance}_{hand_part}.obj'))
            check_obj_path_list.append(hand_path)
            check_new_name_list = []
            check_new_name_list.append(f'{instance}_{1 - hand_part}')
           # check_new_name_list.append(f'{instance}_{hand_part}')
            check_new_name_list.append(f'{instance}_{hand_file[:-4]}')
            env = BaseEnv(infos.render_path, args.sample_num, args.category, args.seq)
            env.sample_and_render(new_urdf_save_path, grasp_path, check_obj_path_list, check_new_name_list, hand_part, instance, int(hand_file[2:-4]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample_num', type=int, default=20, help='each grasp try s times. At most take 10 pictures.')
    parser.add_argument('-c', '--category', type=str, default='laptop')
    parser.add_argument('--scale', type=int, default=1000)
    parser.add_argument('--seq', action='store_true')
    parser.add_argument('-n', '--num_workers', type=int, default=8)
    args = parser.parse_args()
    infos = global_info()

    if args.category == 'laptop':
        total_part = 2

    obj_folder = pjoin(infos.partnet_path, args.category)                       #.urdf
    instance_lst = os.listdir(obj_folder)

    if args.seq:
        test_split_path = pjoin(infos.render_path, 'splits', args.category, 'single_frame', 'test.txt')
        with open(test_split_path, 'r') as f:
            test_split = f.readlines()
        test_ins_list = list(set([i.split('_')[0] for i in test_split]))
        instance_lst = [i for i in instance_lst if i in test_ins_list]

    #TODO: multi processes!
    per_worker_obj_list = []
    length = len(instance_lst) // args.num_workers + 1
    start = 0
    end = length
    for i in range(args.num_workers):
        if i == args.num_workers - 1:
            per_worker_obj_list.append(instance_lst[start:])
        else:
            per_worker_obj_list.append(instance_lst[start: end])
        start += length
        end += length
    process_lst = []
    for proc_num in range(args.num_workers):
        p = Process(target=proc_work,
                    args=(per_worker_obj_list[proc_num], proc_num, infos, args))
        p.start()
        process_lst.append(p)

    for p in process_lst:
        p.join()
