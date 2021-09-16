#!/bin/sh
#SBATCH --job-name=sat-chall
# One hour
#SBATCH --time=01:00:00
#SBATCH --nodes=1
# Where to write console/error logs:
#SBATCH -o logs/slurm/logs-%j
#SBATCH -e logs/slurm/errors-%j
#SBATCH --ntasks=1
# For debug: RMC-C01-DEBUG
#SBATCH --partition=RMC-C01-BATCH
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4


# Retrieve first and second argument passed to the script
LOCAL=$1
LOGDIR=$2
NUM_WORKERS=$3
BATCH_SIZE=$4
GPUS=$5
MAX_EPOCHS=$6



echo "=====Job Infos ===="
echo "Node List: " $SLURM_NODELIST
echo "jobID: " $SLURM_JOB_ID
echo "Partition: " $SLURM_JOB_PARTITION
echo "Submit directory:" $SLURM_SUBMIT_DIR
echo "Submit host:" $SLURM_SUBMIT_HOST
echo "In the directory: `pwd`"
echo "As the user: `whoami`"
echo "Python version: `python -c 'import sys; print(sys.version)'`"
echo "pip version: `pip --version`"
echo "TF version: `python -c 'import tensorflow as tf; print(tf.__version__)'`"


nvidia-smi

echo "local" $LOCAL
echo "logdir:" $LOGDIR
echo "num_workers" $NUM_WORKERS
echo "batch_size" $BATCH_SIZE 
echo "gpus" $GPUS
echo "max_epochs" $MAX_EPOCHS

start_time=`date +%s`
echo "Job Started at "`date`

# Launch your python script
python train.py --local $LOCAL --logdir $LOGDIR --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE --gpus $GPUS --max_epochs $MAX_EPOCHS

echo "Job ended at "`date`
end_time=`date +%s`
total_time=$((end_time-start_time))

echo "Took " $total_time " s"

