import os
import sys
import shlex
def main():
    fixed_args_str = '''../../../mnt/workspace/user/jiexuan.ls/Qwen2.5-VL/qwen_vl_finetune/qwenvl/train/train_causal_star_ori.py --model_name_or_path ../../../mnt/workspace/user/jiexuan.ls/models/Qwen/Qwen2-VL-7B-Instruct  --tune_mm_vision False  --tune_mm_llm True  --tune_mm_mlp True --dataset_use casigcausalrewardstar%100 --output_dir ../../../mnt/workspace/user/jiexuan.ls/Qwen2.5-VL/qwen_vl_finetune/checkpoints/casig_star_causal_reward_lr_5e6_3e5 --run_name casig_star_causal_reward_lr_5e6_3e5 --gradient_checkpointing True --bf16 True --per_device_train_batch_size 4  --per_device_eval_batch_size 1 --gradient_accumulation_steps 4 --learning_rate 5e-6 --mm_projector_lr 3e-5 --vision_tower_lr 1e-6  --optim adamw_torch --model_max_length 8192 --data_flatten True --max_pixels 131072  --min_pixels 4096  --base_interval 2 --video_max_frames 8 --video_min_frames 4  --video_max_frame_pixels 1304576  --video_min_frame_pixels 200704  --num_train_epochs 3 --lr_scheduler_type "cosine" --warmup_ratio 0.1 --weight_decay 0.01 --max_grad_norm 1  --logging_steps 200  --save_strategy "steps" --eval_strategy "no"  --save_steps 2000  --save_total_limit 1 --deepspeed ../../../mnt/workspace/user/jiexuan.ls/Qwen2.5-VL/qwen_vl_finetune/scripts/zero3_star.json --report_to wandb'''
    expanded_args = shlex.split(os.path.expandvars(fixed_args_str))
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["GLOO_SOCKET_IFNAME"] = "bond0"
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", "12355")    
    
    torch_dist_cmds = [f"--nproc_per_node={4}", f"--master_addr={master_addr}", f"--master_port={master_port}"]    
    launch_cmd = "-u -m torch.distributed.run"
    launch_cmds = launch_cmd.split(" ")
    launch_cmds.extend(torch_dist_cmds)    
    
    print(f"sys.executable {sys.executable}")
    print(f"Torchrun Lauch CMDSs: {launch_cmds}")
    
    os.execl(sys.executable, sys.executable, *launch_cmds, *expanded_args)
    
    

if __name__ == '__main__':
    main()