#!/usr/bin/env bash

trait=$1
workPATH="/FULL/WORKING/PATH"

mapfile -t S < ${workPATH}sets/snp_sizes.txt
#specifying memory and time requirements for SLURM submission
mems=(16 16 16 16 16 32 32 64 64 128 256)
tims=(00:59:00 00:59:00 00:59:00 00:59:00 00:59:00 00:59:00 00:59:00 00:59:00 02:59:00 5:59:00 11:59:00)

tmp=${#S[@]}

for s in $(seq 0 $((tmp-1))); do
    sbatch --mem=${mems[$s]}G --time=${tims[$s]} perc-cov.sh $trait ${S[$s]}
done
wait
