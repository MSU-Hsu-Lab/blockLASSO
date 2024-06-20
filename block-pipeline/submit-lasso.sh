#!/usr/bin/env bash

trait=$1
gwas='CACO' #specify if case control or continuous values were used for computing GWAS
workPATH="FULL/PATH/TO/WORKING/DIRECTORY/"

#number of SNVs per block
S=(10 23 50 100 227 500 1000 2273 5000 10000 22726)
#memory requests per block size
mems=(4 4 4 8 8 16 16 32 64 128 256)
#job time limit per block size
tims=(00:59:00 00:59:00 00:59:00 00:59:00 00:59:00 00:59:00 00:59:00 00:59:00 02:59:00 5:59:00 11:59:00)
for s in "${S[@]}";do
    for c in {1..22}; do #loop over chromosomes (blocks)
        sbatch --mem=${mems[$s]}G --time=${tims[$s]} lasso.sh $trait ${S[$s]} $c $gwas
    done
done
wait
