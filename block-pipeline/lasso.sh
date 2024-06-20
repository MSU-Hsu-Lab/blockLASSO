#!/usr/bin/env bash
#SBATCH -n 2 -c 20
#SBATCH --job-name=do-lasso
#SBATCH --output=%x-%a-%A.SLURMout
#SBATCH -a 1-5

# note hard paths have been changed.
module purge
module load GCC/8.3.0 OpenMPI/3.1.4 imkl/2019.5.281
module load Python/3.8.3
source '/mnt/home/rabentim/programs/pysnptools/bin/activate'
k=$SLURM_ARRAY_TASK_ID
echo $k

traitname=$1
m=$2
top=$3
chrome=$4

OUTDIR='/mnt/home/rabentim/methods'/$traitname/
#OUTDIR='/mnt/home/rabentim/methods/aou'/$traitname/
#OUTDIR='/mnt/home/rabentim/snp-projection'/$traitname/
#OUTDIR='/mnt/home/rabentim/metric-projection'/$traitname/
mkdir -p $OUTDIR

genoPATH='/mnt/research/UKBB/hsuGroup/ukb500/genotypes/calls.merged/ukb500.calls.onlyqc'
#genoPATH='/mnt/scratch/rabentim/taiwan/twb2calls-ukbimp/ukb.twn.overlap.qc'
#genoPATH='/mnt/scratch/rabentim/taiwan/twbimp-ukbcalls/ukb.twn.overlap'
#gwasTYPE='CACO'
#gwasTYPE='PCA_YOB_SEX_reg'
gwasTYPE=$5
#covTYPE='YOB_SEX_reg'
covTYPE='PCA_YOB_SEX_reg'
#covTYPE='PCA_YOB_reg'
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
