#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

## SBATCH --ntasks-per-node=1
## SBATCH --time=24:00:00
## SBATCH --job-name=sampling
## SBATCH --mem=10G
## SBATCH --ntasks= ...  # total number of tasks (if not compatible with ntasks-per-node, assigns cores as evenly as possible between workers)

# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
## SBATCH --hint=nomultithread           # 1 processus MPI par coeur physique (pas d'hyperthreading)

#SBATCH --no-requeue
#SBATCH --mail-type=ALL  # email notifications about the job: start, end, ...
## SBATCH --mail-user=pierreantoine.thouvenin@univ-lille.fr

## SBATCH -o /home/username/logs/sampling.out.%A.%N.log
## SBATCH -e /home/username/logs/sampling.err.%A.%N.log

module purge  # remove all previously loaded modules
# module load conda/23.11-py311
# source /share/tools/miniconda3/23.11/bin/activate gpu-dsgs

# * The following is needed to activate conda environment in bash
# https://github.com/conda/conda/issues/7980
# https://askubuntu.com/questions/1218048/activating-conda-environment-in-within-a-shell-script
source ${conda_path} ${env_name}
# source /home/username/miniforge3/bin/activate myenv

# display cpu info
lscpu

# if GPU mode, display GPU info
if [ "$mode" = "gpu" ]
then
    nvidia-smi -q
fi

# run serial SGS sampler
${command} main.py \
--config=${configfile} \
--mode=${mode}

# srun --mpi=pmix_v2 main.py ...

# mamba deactivate
