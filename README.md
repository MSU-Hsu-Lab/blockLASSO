# blockLASSO
Public repository accompanying the publication "Efficient blockLASSO for Polygenic Scores with Applications to All of Us and UK Biobank", containing code and links to predictors. 

## before running
This initial version of the code is focused on
(1) only running and scoring the actual block lasso (i.e., no filtering phenotypes, building testing/training sets, covariate analyses and residual phenotypes, etc.
(2) the blocks are assumed to be entire chromosomes.

# simple submission
This folder contains example scripts to perform the block lasso on a computing cluster. At the very least full paths will have to be defined.
'lasso.sh' and 'submit-lasso.sh' are submission script files. 'ml-single.py' is the actual code to perform the block lasso which implimenents scikit-learn's 'lasso_path'. 'pipeline_utilities.py' contains a large number of helpfully defined functions, however, only one 'read_bed_file' is actually needed for this project.
after definig all relevant paths the scripts can be run via

sh submit-lasso.sh TRAITNAME

These submission scripts assume the use of the SLURM workload manager
