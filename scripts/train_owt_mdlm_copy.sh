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

export WANDB_INIT_TIMEOUT=300


python -u -m main \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  model=small \
  data=openwebtext-split \
  wandb.name=mdlm-owt-356 \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000