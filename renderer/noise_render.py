import argparse
import os
from multiprocessing import Process
import sapien.core as sapien

import sys
from os.path import join as pjoin
import os.path as osp
BASEPATH = osp.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, ".."))

from common.global_info import global_info

class VulkanEnv:
    def __init__(self, save_path, sample_num, category="laptop", seq=False):
        engine = sapien.Engine()

def proc_work(instance_lst, proc_num, infos, args):
    obj_folder = pjoin(infos.partnet_path, args.category)  # .urdf
    hand_folder = pjoin(infos.hand_path, args.category, f'scale{args.scale}')  # .obj
    blender_lst = []
    env = BaseEnv(infos.render_path, args.sample_num, args.category, args.seq)
    
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample_num', type=int, default=20, help='each grasp try s times. At most take 10 pictures.')
    parser.add_argument('-c', '--category', type=str, default='laptop')
    parser.add_argument('--scale', type=int, default=1000)
    parser.add_argument('--seq', action='store_true')
    parser.add_argument('-n', '--num_workers', type=int, default=8)
    return parser.parse_args()

def main():
    args = parse_args()
    infos = global_info()
    category = args.category
    
    if category in ["laptop"]:
        total_part = 2
        
    obj_folder = pjoin(infos.partnet_path, category)
    instance_lst = os.listdir(obj_folder)
    
    if args.seq:
        test_split_path = pjoin(infos.render_path, 'splits', args.category, 'single_frame', 'test.txt')
        with open(test_split_path, 'r') as f:
            test_split = f.readlines()
        test_ins_list = list(set([i.split('_')[0] for i in test_split]))
        instance_lst = [i for i in instance_lst if i in test_ins_list]
        
    per_worker_obj_list = []
    length = len(instance_lst) // args.num_workers
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


if __name__ == "__main__":
    main()
    