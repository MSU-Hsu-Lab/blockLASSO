#!/usr/bin/env bash
#SLURM COMMANDS (timing and memory already specified)

#load modules

traitname=$1
s=$2
echo $traitname
OUTDIR='/FULL/WORKING/PATH'
genoPATH='/PATH/TO/BED/FILES'
cohortDIR='/PATH/TO/COHOR/FILES'
covtype='PCA_YOB_SEX_reg'

python per-chrome-cov.py --geno-path $genoPATH \
        --trait $traitname \
        --cohort-path $cohortDIR \
        --array-id $k \
        --snp-size $s \
        --cov-type $covtype\
        --working-directory $OUTDIR\
        --sex 'both'\
