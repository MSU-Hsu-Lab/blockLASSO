#!/usr/bin/env bash

trait=$1

workPATH="/FULL/WORKING/PATH"

mapfile -t M < ${workPATH}sets/train_sizes.txt
mapfile -t S < ${workPATH}sets/snp_sizes.txt

for m in "${M[@]}"; do
    for s in "${S[@]}"; do
        for c in {1..22}; do
            sbatch score-lasso.sh $trait $m $s $c
        done
    done
done
wait
