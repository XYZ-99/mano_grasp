"""
python ./data_generation/count_grasps.py --grasps-dir XXX
"""
import argparse
import os.path as osp
import glob
from os.path import join as pjoin
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasps-dir", type=str)
    
    return parser.parse_args()

def main():
    args = parse_args()
    grasps_dir = args.grasps_dir
    
    if not osp.exists(grasps_dir):
        raise ValueError("grasps_dir must be provided!")
    
    json_paths = glob.glob(pjoin(grasps_dir, "*.json"))
    
    grasps_count_tuple = []
    for json_path in json_paths:
        json_filewholename = osp.basename(json_path)
        with open(json_path, "r") as f:
            json_content = json.load(f)
        grasps_cnt = len(json_content["grasps"])
        grasps_count_tuple.append((json_filewholename, grasps_cnt))
        
    grasps_count_tuple.sort(key=lambda x: x[1], reverse=True)  # sort by grasps_cnt
    
    for name, cnt in grasps_count_tuple:
        print(f"{name}: {cnt}")
        
        
if __name__ == "__main__":
    main()
    