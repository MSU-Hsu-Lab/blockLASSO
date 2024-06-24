#!/usr/bin/env bash
#SBATCH SPECIFY SLURM COMMANDS (OR OTHER WORKLOAD MANAGER)

#load modules

##########
traitname=$1
echo $traitname
m=$2
s=$3
c=$4
ML='LASSO'
echo $ML
#specify validation set size
valN=2500
OUTDIR='FULL/WOKRING/PATH'

genoPATH='PATH/TO/BED/FILES'
#cohorts (e.g. ancestry groups, sibling groups, etc.) are identified beforehand
cohortDIR='PATH/TO/COHORT/FILES'
gwastype='CACO'
#covariates adjusted for (for file naming conventions)
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
