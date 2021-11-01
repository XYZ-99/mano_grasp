import os
from os.path import join as pjoin
import argparse
import json
import trimesh
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mug-objs-dir",
                        default="/home/xuyinzhen/Documents/mano_grasp/models/mug")
    parser.add_argument("--mapping-txt",
                        default="/home/xuyinzhen/Documents/mano_grasp/models/mug/nocs_mapping.txt",
                        help="Mapping from model index to model id")
    parser.add_argument("--mapping-json",
                        default="/home/xuyinzhen/Documents/mano_grasp/models/mug_handle/model_id_to_handle_path.json",
                        help="Mapping from model id to handle obj path")
    parser.add_argument("--target-dir",
                        default="/home/xuyinzhen/Documents/mano_grasp/models/mug_handle")
    parser.add_argument("--scale",
                        type=float,
                        default=0.25)  # Inherit from category2scale["mug"]
    
    return parser.parse_args()

def main():
    args = parse_args()
    mug_objs_dir = args.mug_objs_dir
    mapping_txt = args.mapping_txt
    mapping_json_path = args.mapping_json
    target_dir = args.target_dir
    scale = args.scale
    
    obj_file_whole_names = os.listdir(mug_objs_dir)
    obj_file_whole_names = list(
        filter(
            lambda name: name.endswith(".obj"),
            obj_file_whole_names
        )
    )
    obj_file_whole_names.sort()
    
    with open(mapping_txt, "r") as f:
        model_ids = f.read().splitlines()
    
    with open(mapping_json_path, "r") as f:
        mapping_model_id_to_handle_path = json.load(f)
    
    for obj_file_whole_name in obj_file_whole_names:
        obj_file_suffix = obj_file_whole_name.split(".")[0]  # e.g., 00001
        obj_idx = int(obj_file_suffix)
        model_id = model_ids[obj_idx]  # e.g., /home/xuyinzhen/Documents/obj_models/train/03797390/d46b98f63a017578ea456f4bbbc96af9
        model_id = osp.basename(model_id)
        handle_path = mapping_model_id_to_handle_path.get(model_id, None)
        
        if handle_path is None:
            continue
        
        """ Copy to destination """
        mesh = trimesh.load(handle_path, force="mesh")
        mesh.vertices = mesh.vertices * scale
        mesh_txt = trimesh.exchange.obj.export_obj(mesh, 
                                                   include_normals=False, 
                                                   include_color=False, 
                                                   include_texture=False,
                                                   return_texture=False, 
                                                   write_texture=False, 
                                                   resolver=None, 
                                                   digits=8)
        
        save_path = pjoin(target_dir, obj_file_whole_name)
        with open(save_path, "w") as f:
            f.write(mesh_txt)
        
        
if __name__ == "__main__":
    main()
