#!/usr/bin/env bash
#SBATCH SPECIFY SLURM COMMANDS (OR OTHER WORKLOAD MANAGER)

#load modules

traitname=$1
m=$2
top=$3
chrome=$4

OUTDIR='/mnt/home/rabentim/methods'/$traitname/

mkdir -p $OUTDIR

genoPATH='/mnt/research/UKBB/hsuGroup/ukb500/genotypes/calls.merged/ukb500.calls.onlyqc'

gwasTYPE=$5
covTYPE='PCA_YOB_SEX_reg'
ML='LASSO'
echo $ML

plinkversion=2
echo 'plink version $plinkversion'

python ml-single.py --geno-path $genoPATH \
    --trait $traitname \
    --cv-fold $k \
    --gwas-type $gwasTYPE\
    --cov-type $covTYPE\
    --ml-type $ML\
    --working-path $OUTDIR \
    --train-size $m \
    --snp-size $top \
    --chrm $chrome \
    --plinktype $plinkversion
