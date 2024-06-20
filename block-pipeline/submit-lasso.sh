#!/usr/bin/env bash

trait=$1
gwas='CACO'
#gwas='PCA_YOB_SEX_reg'
#workPATH="/mnt/home/rabentim/metric-projection/${trait}/"
workPATH="/mnt/home/rabentim/methods/${trait}/"
mapfile -t M < ${workPATH}sets/train_sizes.txt
#mapfile -t S < ${workPATH}sets/snp_sizes.txt
#S=(10 23 50 100 227 500 1000 2273 5000 10000 22726)
S=(10 23 50 100 227 500 1000 2273 5000)
mems=(4 4 4 8 8 16 16 32 64 128 256)
tims=(00:59:00 00:59:00 00:59:00 00:59:00 00:59:00 00:59:00 00:59:00 00:59:00 02:59:00 5:59:00 11:59:00)
for m in "${M[@]}"; do
    for s in {0..8}; do
#    for s in {7..7};do
#    for s in "${S[@]}";do
#    for s in {7..7};do
#     for s in 10 23 50 100 227 500; do
#     for s in 1000 2273 5000 10000 22726; do
        for c in {1..22}; do
            sbatch --mem=${mems[$s]}G --time=${tims[$s]} lasso.sh $trait $m ${S[$s]} $c $gwas
        done
    done
done
wait

#sbatch lasso.sh
