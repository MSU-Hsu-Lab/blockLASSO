#!/usr/bin/env bash

trait=$1
workPATH="/FULL/WORKING/PATH"
traits=("LIST" "OF" "PHENOTYPE" "NAMES")

for trait in ${traits[@]}; do
    sbatch cov.sh $trait
done
wait
