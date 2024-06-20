#!/usr/bin/env bash
#SBATCH -n 1 -c 20
#SBATCH --time=6:59:00
#SBATCH --mem=256G
#SBATCH --job-name=make-covariance
#SBATCH --output=%x-%a-%A.SLURMout
#SBATCH -a 1-5

module purge
module load GCC/8.3.0 OpenMPI/3.1.4 imkl/2019.5.281
module load Python/3.8.3
source '/mnt/home/rabentim/programs/pysnptools/bin/activate'
k=$SLURM_ARRAY_TASK_ID
echo $k

traitname=$1
s=$2
echo $traitname
OUTDIR='/mnt/home/rabentim/methods'/$traitname/
genoPATH='/mnt/research/UKBB/hsuGroup/ukb500/genotypes/calls.merged/ukb500.calls.onlyqc'
cohortDIR='/mnt/research/UKBB/hsuGroup/ukb500/cohorts/'
covtype='PCA_YOB_SEX_reg'

python per-chrome-cov.py --geno-path $genoPATH \
        --trait $traitname \
        --cohort-path $cohortDIR \
        --array-id $k \
        --snp-size $s \
        --cov-type $covtype\
        --working-directory $OUTDIR\
        --sex 'both'\
