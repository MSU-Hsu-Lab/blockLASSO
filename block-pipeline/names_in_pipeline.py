import re
import os
from functools import reduce

# Module to handle file names for the pipeline
# NB: the following names are might not be taken from here somewhere in
# pipeline
# - column names in phen files
# - gwas names are defined in shell script for gwas

# ----- Filename constants -----
DATA_FILE_EXT = ".csv"
GENO_BASENAME = \
"/mnt/research/UKBB/hsuGroup/ukb500/genotypes/calls.merged/ukb500.calls.onlyqc"
PLINK_SCRIPT = "/mnt/home/lellolou/programs/plink1.9/plink"

GWAS_BASE = "gwas."
GWAS_STUDY_PHEN_PLACEHOLDER_NAME = "gwasPhen.{trait}.{cohort}_{index}.csv"
GWAS_STUDY_PLACEHOLDER_NAME = "gwas.{trait}.{cohort}_{index}"
JOB_NAME = "lasso.{}"
LASSO_BETAS_BASE = "lasso.betas."
LASSO_LAMBDAS_BASE = "lasso.lambdas."
LASSO_TOPBETAS_BASE = "lasso.topBetas."
LASSO_PHEN_SCALING_BASE = "lasso.phen.scaling."
LASSO_PHEN_SCALING_TOPBETAS_BASE = "lasso.phen.scaling.topBetas."
PHEN_BASE = "phen."
PHEN_COHORT_PLACEHOLDER_NAME = PHEN_BASE + "{trait}.{cohort}_{jobid}{ext}"
PHEN_NON_PHEN_COLS = ["EID", "YOB", "SEX"]
PHEN_CV_PLACEHOLDER_NAME = PHEN_BASE  \
    + "{trait}.{cohort}.CV{cv_idx}{trainval}_{jobid}{ext}"
PHEN_FULL_PLACEHOLDER_NAME = PHEN_BASE + "{trait}_{jobid}{ext}"
PHEN_HOLD_OUT_PLACEHOLDER_NAME = PHEN_BASE \
    + "{trait}.{cohort}.HO_{jobid}{ext}"
PRED_BASE = "pred."
PRED_TOPBETAS_BASE = PRED_BASE + "topBetas."
REPORT_BASE = "runReport."
REPORT_EXT = ".log"
REPORT = REPORT_BASE[:-1] + REPORT_EXT
SCORES_BASE = "scores."
SCORES_TOPBETAS_BASE = "scores.topBetas."

# ----- Writing and reading data formats -----
FORMAT_BETAS_READ_CSV = {"sep": r"\s+", "header": 0}
FORMAT_BIM_COLUMNS = ['chrom', 'snp', 'cm', 'pos', 'a0', 'a1']
FORMAT_BIM_READ_CSV = {
    "sep": r"\s+",
    "header": None,
    "names": FORMAT_BIM_COLUMNS
}
FORMAT_GWAS_READ_CSV = {"sep": r"\s+", "header": 0}
FORMAT_GWAS_TO_CSV = {"sep": r" ", "header": False, "index": False}
FORMAT_LASSO_TO_CSV = {"sep": r" ", "index": False}
FORMAT_PHEN_CASE_COLUMNS = ["EID", "YOB", "SEX", "CC", "ONSET"]
FORMAT_PHEN_CONT_COLUMNS = ["EID", "YOB", "SEX", "PHEN"]
FORMAT_PHEN_READ_CSV = {"sep": r"\s+", "header": 0}
FORMAT_PHEN_TO_CSV = {"sep": " ", "index": False, "header": True}
FORMAT_PHEN_SCALING_READ_CSV = {"sep": r"\s+", "header": 0}
FORMAT_PHEN_SCALING_TO_CSV = {"sep": " ", "index": False, "header": True}
FORMAT_SCORES_TO_CSV = {"sep": " ", "index": False, "header": True}
FORMAT_SCORES_READ_CSV = {"sep": " ", "header": 0}

FORMAT_MV_LASSO_PHEN_SCALING_TO_CSV = \
    {"sep": " ", "index": True, "header": True}
FORMAT_MV_LASSO_PHEN_SCALING_READ_CSV = \
    {"sep": " ", "index_col": 0, "header": 0}


def trim_cohort_filename_to_cohort_name(filename):
    '''Remove file extension from cohort file name.'''
    name = re.sub(r"\.(txt|csv|fam)$", "", filename)
    return name


def get_cohort_fileextension(string):
    '''Check if string is filename by looking for file extensions. Returns file
    extension if found, otherwise None.'''
    match = re.search(r"\.(txt|csv|fam)$", string)
    return bool(match)


def gwas_basename_from_phen_path_or_file(phen_name):
    name = re.sub("^.*/?" + PHEN_BASE, GWAS_BASE, phen_name)
    name = re.sub(r"(.*)\.(csv|txt)", r"\1", name)
    return name


def gwas_path_from_phen_path(phen_path, gwas_ext):
    name = sub_base(PHEN_BASE, GWAS_BASE, phen_path)
    name = re.sub(r"(.*)\.(csv|txt)$", r"\1" + gwas_ext, name)
    return name


def filename_from_phen(phen_path_or_name):
    name = re.sub(r".*{}(.*\.CV\d.*)(\.(txt|csv))".format(PHEN_BASE), r"\1\2",
                  phen_path_or_name)
    return name


def train_scaling_path_from_phen_path(train_phen_path):
    dir_string = re.sub(r"(.*)phen\.(.*\.CV\d.*)(\.(txt|csv))", r"\1",
                        train_phen_path)
    name = re.sub(r".*phen\.(.*\.CV\d.*)(\.(txt|csv))",
                  r"lasso.phen.scaling.\1.csv", train_phen_path)
    return dir_string + name


def test_phen_path_from_train_phen_path(train_phen_path):
    path = re.sub(r"(.*phen\..*\.CV\d+)train", r"\1valid", train_phen_path)
    return path


def train_phen_path_from_test_phen_path(test_phen_path):
    path = re.sub(r"(.*phen\..*\.CV\d+)valid", r"\1train", test_phen_path)
    return path


def phen_path_from_CV_string(cv_string, directory):
    pattern = PHEN_BASE + ".*" + cv_string
    phen_filename = next(fn for fn in os.listdir(directory)
                         if re.match(pattern, fn))
    phen_path = directory + phen_filename
    return phen_path


def scores_name_from_phen_path(train_phen_path, test_phen_path):
    # Check if test_phen is "validation version" of train phen
    if test_phen_path == \
            test_phen_path_from_train_phen_path(train_phen_path):
        name = re.sub(r".*phen\.(.*\.CV\d+).*(_\d+)\.(txt|csv)",
                      r"scores.\1\2.csv", train_phen_path)
    else:
        name_from_train = re.sub(r".*phen\.(.*\.CV\d.*)(\.(txt|csv))",
                                 r"scores.\1", train_phen_path)
        name_from_test = re.sub(r".*phen\.(.*\.CV\d.*)(\.(txt|csv))", r"\1",
                                test_phen_path)
        name = name_from_train + "_on_" + name_from_test + ".csv"

    return name


def betas_path_from_train_phen_path(train_phen_path):
    dir_string = re.sub(r"(.*)phen\.(.*\.CV\d.*)(\.(txt|csv))", r"\1",
                        train_phen_path)
    name = re.sub(r".*phen\.(.*\.CV\d.*)(\.(txt|csv))", r"lasso.betas.\1",
                  train_phen_path)
    name = dir_string + name + ".csv"
    return name


def betas_path_from_scores_path(score_path):
    betas_path = sub_base(SCORES_BASE, LASSO_BETAS_BASE, score_path)
    betas_path = re.sub(r"CV\d+", r"\g<0>train", betas_path)
    return betas_path


def CV_ranges_from_file_list(file_list):
    k_folds = [int(re.match(r".*CV(\d+).*?", fn)[1]) for fn in file_list]
    k_folds.sort()

    def sequencify(list_of_seqs, new):
        if not list_of_seqs:
            return [(new, new)]
        last = list_of_seqs[-1]
        if new - last[-1] > 1:
            list_of_seqs.append((new, new))
        else:
            list_of_seqs[-1] = (last[0], new)
        return list_of_seqs

    seqs = reduce(sequencify, k_folds, [])
    ret = ""
    for seq in seqs:
        if seq[0] == seq[1]:
            ret += f"{seq[0]}."
        else:
            ret += f"{seq[0]}-{seq[1]}."
    return ret[:-1]


def top_betas_filename_from_file_list(file_list, base_to_sub):
    file_name = sub_base(base_to_sub, LASSO_TOPBETAS_BASE, file_list[0])
    file_name = re.sub(r"CV[0-9a-zA-Z]+_",
                       f"CV{CV_ranges_from_file_list(file_list)}_", file_name)
    return file_name


def phenotype_from_beta_name(beta_name):
    return re.sub(r"beta_(.*?)(_CV\d+)?(_|\.)\d+$", r"\1\2", beta_name)


def all_files_in_dir_containing(d, pattern):
    files = [
        fn for fn in os.listdir(d)
        if os.path.isfile(os.path.join(d, fn)) and re.search(pattern, fn)
    ]
    return sorted(files)


# def report_name_for_lasso_fold(trunk):
#     return REPORT_BASE + trunk + REPORT_EXT


def report_name_for_lasso_fold_from_phen_path(phen_path):
    name = filename_from_path(phen_path)
    name = re.sub("^" + PHEN_BASE, REPORT_BASE, name)
    name = re.sub(r"\.(csv|txt)$", REPORT_EXT, name)
    return name


def dir_from_path(path):
    return os.path.dirname(path) + "/"


def filename_from_path(path):
    return os.path.basename(path)


def sub_base(old_base, new_base, file_name):
    '''Substitute the base of a file name.'''
    name = re.sub(r"(^.*/)?" + old_base, r"\1" + new_base, file_name)
    return name
