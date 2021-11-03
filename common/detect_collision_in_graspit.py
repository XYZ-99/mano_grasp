import os
from os.path import join as pjoin
import trimesh
from geometry_msgs.msg import Pose
import transforms3d as tf
import numpy as np

import sys
sys.path.append(pjoin(__file__, ".."))

from mano_grasp.graspit_process import GraspitProcess


def check_collision(obj_path_list, obj_pose_list, new_name_list):
    xml_path_list = []
    for i in range(len(obj_path_list)):
        xml_path_list.append(save_file_for_graspit(obj_path_list[i], new_name_list[i]))
    proccess = GraspitProcess()

    if not proccess.run:
        proccess.start()
    graspit = proccess.graspit
    graspit.clearWorld()
    for i in range(len(xml_path_list)):
        graspit.importGraspableBody(xml_path_list[i], pose=mat_to_Pose(obj_pose_list[i]))
    #graspit.toggleAllCollisions(True)
    x = graspit.noCollision()
    #time.sleep(10)
    return x


def save_file_for_graspit(file_path, new_name):
    mesh = trimesh.load(file_path, force='mesh')
    new_off_name = new_name + '.off'
    save_off_path = pjoin(os.environ['GRASPIT'], 'models', 'objects', new_off_name)
    m = trimesh.exchange.export.export_mesh(mesh, 
                                            file_obj=save_off_path, 
                                            file_type='off')

    if not m:
        print(f'fail to save {save_off_path}')
        return False

    new_xml_name = new_name + '.xml'
    save_xml_path = pjoin(os.environ['GRASPIT'], 'models', 'objects', new_xml_name)
    with open(save_xml_path, 'w') as f:
        f.write(f'<root><geometryFile type="off">{new_off_name}</geometryFile></root>')
    return new_xml_name[:-4]

def RT_to_Pose(R, T):
    pose = Pose()
    pose.position.x = T[0]/1000
    pose.position.y = T[1]/1000
    pose.position.z = T[2]/1000

    q = tf.quaternions.mat2quat(R)
    pose.orientation.w = q[0]
    pose.orientation.x = q[1]
    pose.orientation.y = q[2]
    pose.orientation.z = q[3]
    return pose

def mat_to_Pose(mat):
    pose = Pose()
    pose.position.x = mat[0, 3]/1000
    pose.position.y = mat[1, 3]/1000
    pose.position.z = mat[2, 3]/1000

    q = tf.quaternions.mat2quat(mat[:3, :3])
    pose.orientation.w = q[0]
    pose.orientation.x = q[1]
    pose.orientation.y = q[2]
    pose.orientation.z = q[3]
    return pose


if __name__ == "__main__":
    obj_path = ['/home/hewang/Desktop/data/jiayi/h2o_data/hands/laptop/scale300/12115/1_0.obj',
                '/home/hewang/Desktop/data/jiayi/h2o_data/objs/laptop/12115_0.obj',
                '/home/hewang/Desktop/data/jiayi/h2o_data/objs/laptop/12115_1.obj']
    new_name = ['12115_1_0','12115_0','12115_1']
    pose0 = np.eye(4)
    pose1 = np.eye(4)
    pose1[:3,3] = [-0.45497730734333164, 0.2, 0.21]
    pose_list = [pose1, pose0, pose1]
    if check_collision(obj_path,pose_list, new_name):
        print('no collision')
    else:
        print('collision!')
