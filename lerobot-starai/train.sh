python lerobot/scripts/train.py \
  --dataset.repo_id=starai/starai3\
  --policy.type=act \
  --output_dir=outputs/train/act_starai2 \
  --job_name=act_starai \
  --policy.device=cuda \
  --wandb.enable=false \
  --dataset.video_backend=pyav