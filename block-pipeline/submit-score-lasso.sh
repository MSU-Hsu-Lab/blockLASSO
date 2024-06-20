#!/usr/bin/env bash

trait=$1
#workPATH="/mnt/home/rabentim/metric-projection/${trait}/"
workPATH="/mnt/home/rabentim/methods/${trait}/"
#workPATH="/mnt/home/rabentim/snp-projection/${trait}/"
mapfile -t M < ${workPATH}sets/train_sizes.txt
#mapfile -t S < ${workPATH}sets/snp_sizes.txt
S=(10 23 50 100 227 500 1000 2273 5000)
#S=(2273)

for m in "${M[@]}"; do
    for s in "${S[@]}"; do
#    for s in 2273; doi
#    for s in 10 23 50 100 227 500; do
#    for s in 10000 22726; do 
        for c in {1..22}; do
#	for c in 10 17; do
            sbatch score-lasso.sh $trait $m $s $c
        done
    done
done
wait
