import os
import sys
import shlex

def main():
    print('cwd: ', os.getcwd())
    print('notebook lauch!')

    # os.chdir("/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/DiffusionRL/src")

    dir = '/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/DiffusionRL'
    files = os.listdir(dir)
    files = [f for f in files if os.path.isfile(os.path.join(dir, f))]

    print(files)

    fixed_args_str = '''/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/DiffusionRL/src/preprocess_flux_embedding.py \
        --output_dir /mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/DiffusionRL/data/rl_embeddings \
        --prompt_csv /mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/Data/diffusion_rl/csv/train.csv'''
    expanded_args = shlex.split(os.path.expandvars(fixed_args_str))
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", "12355")    
    
    torch_dist_cmds = [f"--nproc_per_node={1}", f"--master_addr={master_addr}", f"--master_port={master_port}"]    
    launch_cmd = "-u -m torch.distributed.run"
    launch_cmds = launch_cmd.split(" ")
    launch_cmds.extend(torch_dist_cmds)    
    
    print(f"sys.executable {sys.executable}")
    print(f"Torchrun Lauch CMDSs: {launch_cmds}")
    
    os.execl(sys.executable, sys.executable, *launch_cmds, *expanded_args)
    
    

if __name__ == '__main__':
    main()