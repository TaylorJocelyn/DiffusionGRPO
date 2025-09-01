GPU_NUM=2 # 2,4,8
MODEL_PATH="/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/.cache/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
OUTPUT_DIR="data/rl_embeddings"

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    src/preprocess_flux_embedding.py \
    --output_dir $OUTPUT_DIR \
    --prompt_csv "/mnt/workspace/user/zengdawei.zdw/shared/jinghan.zq/Data/diffusion_rl/csv/train.csv"
