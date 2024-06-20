#!/usr/bin/env bash
#SBATCH -n 1 -c 10
#SBATCH --time=3:59:00
#SBATCH --mem=256G
#SBATCH --job-name=outscore
#SBATCH --output=%x-%a-%A.SLURMout
#SBATCH -a 1-5

module purge
module load GCC/8.3.0 OpenMPI/3.1.4 imkl/2019.5.281
module load Python/3.8.3
source '/mnt/home/rabentim/programs/pysnptools/bin/activate'
k=$SLURM_ARRAY_TASK_ID
echo $k

traitname=$1
echo $traitname
m=$2
s=$3
c=$4
ML='LASSO'
echo $ML
valN=2500
#valN=250
OUTDIR='/mnt/home/rabentim/methods'/$traitname/
#OUTDIR='/mnt/home/rabentim/snp-projection'/$traitname/
#OUTDIR='/mnt/home/rabentim/metric-projection'/$traitname/
#OUTDIR='/mnt/home/rabentim/taiwan/qc'/$traitname/
genoPATH='/mnt/research/UKBB/hsuGroup/ukb500/genotypes/calls.merged/ukb500.calls.onlyqc'
#genoPATH='/mnt/scratch/rabentim/taiwan/twb2calls-ukbimp/ukb.twn.overlap.qc'
cohortDIR='/mnt/research/UKBB/hsuGroup/ukb500/cohorts/'
gwastype='CACO'
#gwastype='PCA_YOB_SEX_reg'
#covtype='PCA_YOB_reg'
#covtype='YOB_SEX_reg'
#covtype='PCA_YOB_reg'
covtype='PCA_YOB_SEX_reg'

python score-ML.py --geno-path $genoPATH \
        --trait $traitname \
        --cohort-path $cohortDIR \
        --array-id $k \
        --val-size $valN\
        --ml-type $ML\
        --gwas-type $gwastype\
        --cov-type $covtype\
        --working-directory $OUTDIR\
        --sex 'both'\
        --print-score 'yes' \
        --train-size $m \
        --snp-size $s \
        --chrm $c
