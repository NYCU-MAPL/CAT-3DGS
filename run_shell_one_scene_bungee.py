import os
from argparse import ArgumentParser, Namespace
import sys

s_pth="path/to/your/datasets/db"

parser = ArgumentParser(description="Training script parameters")
parser.add_argument("-s", "--scene", type=str, default="amsterdam", help="Scene to train on")
args = parser.parse_args(sys.argv[1:])


for lmbda in [0.002]: # [0.001, 0.002, 0.003, 0.006, 0.01, 0.02]
    scene = args.scene
    project_name=f"CAT3DGS_db_{scene}"
    txt_name=f"output_result/db/{scene}"
    m_pth=f"output_log/db"
    
    one_cmd = f'python train.py --eval --lod 30 --voxel_size 0 --update_init_factor 128 --iterations 40_000 --test_iterations 40_000 -s {s_pth}/{scene} -m {m_pth}/{scene}/{lmbda} --exp_name {scene}_l{lmbda} --lmbda {lmbda} --project_name {project_name} --cam_mask 25 --txt_name {txt_name}  --chcm_slices_list 5 10 15 20 --config ./arguments/bungee.py'
    os.system(one_cmd)
