#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=aw624 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH
# the above path could also point to a miniconda install
# if using miniconda, uncomment the below line
source ~/.bashrc
source /vol/bitbucket/${USER}/miniconda3/etc/profile.d/conda.sh
# source /vol/cuda/12.0.0/setup.sh
conda activate mdlm

/usr/bin/nvidia-smi
uptime

# python -u -m main \
#   loader.batch_size=512 \
#   loader.eval_batch_size=512 \
#   model=tiny \
#   data=lm1b \
#   parameterization=subs \
#   model.length=128 \
#   sampling.steps=1000 \
#   model.low_rank_attn=True \
#   model.low_rank_percentage=0.9 \
#   model.timestep_low_rank=True \
#   model.low_rank_mode=learnable \
#   model.learnable_gate_mode=hard_ste \
#   training.flops_lambda=1 \
#   model.learnable_gate_threshold=0.05 \
#   training.target_activation_ratio=0.55 \
#   wandb.name=mdlm-lm1b-NALR-learnable-timestep-hard_ste_0.5_1

python -u -m main \
  loader.batch_size=512 \
  loader.eval_batch_size=512 \
  model=tiny \
  data=lm1b \
  parameterization=subs \
  model.length=128 \
  sampling.steps=1000 \
  model.low_rank_attn=True \
  model.low_rank_percentage=0.9 \
  model.timestep_low_rank=True \
  model.low_rank_mode=frozen_schedule \
  model.learnable_gate_mode=hard_ste \
  model.learnable_gate_threshold=0.05 \
  training.frozen_schedule_ckpt=/vol/bitbucket/aw624/mdlm/outputs/lm1b/2026.02.26/092548/checkpoints/last.ckpt \
  wandb.name=mdlm-lm1b-NALR-frozen-schedule-hard_ste_0.5