import os
from os.path import join as pjoin
import argparse
import json
import os.path as osp
import trimesh


def check_partnet_mug_path(path):
    if "partnet_mug" in os.listdir(path):
        return pjoin(path, "partnet_mug")
    else:
        return path
    
def find_which_is_handle(dir_path, obj_list):
    # handle's com should have the max z
    max_z = -10
    max_z_name = None
    for obj_wholename in obj_list:
        obj_path = pjoin(dir_path, obj_wholename)
        mesh = trimesh.load(obj_path)
        center_mass = mesh.center_mass
        if center_mass[2] > max_z:
            max_z = center_mass[2]
            max_z_name = obj_wholename
    
    return max_z_name
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partnet-mug-path", 
                        default="/home/xuyinzhen/Downloads/partnet_mug/partnet_mug")
    parser.add_argument("--json-save-path", 
                        default="/home/xuyinzhen/Documents/mano_grasp/models/mug_handle/model_id_to_handle_path.json")
    
    return parser.parse_args()

def main():
    args = parse_args()
    partnet_mug_path = check_partnet_mug_path(args.partnet_mug_path)
    json_save_path = args.json_save_path

    dir_names = os.listdir(partnet_mug_path)
    dir_names = list(
        filter(
            lambda p: p[0] != "." and p.isdigit(),
            dir_names
        )
    )
    
    mapping_from_model_id_to_handle_path = {}
    model_cnt = {}
    dir_names.sort()
    for dir_name in dir_names:
        dir_path = pjoin(partnet_mug_path, dir_name)
        meta_json_path = pjoin(dir_path, "meta.json")
        with open(meta_json_path, "r") as f:
            json_data = json.load(f)
            
        model_id = json_data["model_id"]
        handle_dir_path = pjoin(dir_path, "objs")
        if not osp.exists(handle_dir_path):
            # some have no objs dir
            continue
        
        obj_wholenames = os.listdir(handle_dir_path)
        obj_wholenames.sort()
        handle_wholename = find_which_is_handle(handle_dir_path, obj_wholenames)
        handle_path = pjoin(handle_dir_path, handle_wholename)
        mapping_from_model_id_to_handle_path[model_id] = handle_path
        model_cnt[model_id] = model_cnt.get(model_id, 0) + 1

    # for k, v in model_cnt:
    #     if v != 1:
    #         print(f"{k}: {v}")
    # print(model_cnt)
    # TODO: Multiple mapping for one instance! (up to 3)
        
    with open(json_save_path, "w") as f:
        json.dump(
            mapping_from_model_id_to_handle_path,
            f
        )

if __name__ == "__main__":
    main()
