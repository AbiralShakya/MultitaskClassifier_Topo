#!/bin/sh

#SBATCH --job-name=mat_topo_train       
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1           
#SBATCH --cpus-per-task=16            
#SBATCH --mem=128G                     
#SBATCH --time=5:00:00                
#SBATCH --gres=gpu:1                    
#SBATCH --output=slurm_job_%j.out      
#SBATCH --error=slurm_job_%j.err        
#SBATCH --mail-type=BEGIN,END,FAIL     
#SBATCH --mail-user=as0714@princeton.edu

module purge
module load anaconda3/2024.10 


# module load cudatoolkit/11.8
# module load cudnn/8.9.2.26_cuda11.8

source ~/.bashrc
conda activate topological_ml 

cd /scratch/gpfs/as0714/graph_vector_topological_insulator/multitask_ti_classification/src

echo "Starting Python training script at $(date)"
python main.py

echo "Python training script finished at $(date)"