#!/usr/bin/env bash

trait=$1
workPATH="/mnt/home/rabentim/metric-projection/${trait}/"
#workPATH="/mnt/home/rabentim/sparse-methods/${trait}/"
#workPATH="/mnt/home/rabentim/snp-projection/${trait}/"
traits=("gout" "hgt" "CAD" "BMI" "asthma" "diabetes.type1" "diabetes.type2" "hypertension" "hyperlip-cont" "Direct.bilirubin" "Lipoprotein.A" "atrial.fibrillation")

for trait in ${traits[@]}; do
    sbatch cov.sh $trait
done
wait

#sbatch lasso.sh
