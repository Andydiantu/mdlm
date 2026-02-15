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

python -u -m main \
  loader.batch_size=512 \
  loader.eval_batch_size=512 \
  model=tiny \
  data=lm1b \
  parameterization=subs \
  model.length=128 \
  sampling.steps=1000 \
  model.low_rank_attn=True \
  model.low_rank_percentage=0.5 \
  wandb.name=mdlm-lm1b
