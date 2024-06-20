from time import time
import re
import sys
import numpy as np
import pandas as pd
import names_in_pipeline as nip
from pysnptools.snpreader import Bed


# read phen file
def read_csv_infer_if_header(fname, read_kws, columns=None):
    read_kws = {k: read_kws[k] for k in read_kws.keys() - {"header"}}
    # read first rows of file to check if header
    df = pd.read_csv(fname, header=None, **read_kws, nrows=4)
    # check if all elements on first row contains x letters in a row
    is_col_names = df.iloc[0, :]\
        .apply(lambda x: re.search(r"[a-zA-Z]{1,}", str(x)))
    if all(is_col_names):
        df = pd.read_csv(fname, header=0, **read_kws)
    else:
        df = pd.read_csv(fname, header=None, **read_kws)

    if columns:
        df.columns = columns
    return df


# Convenience methods for reading csv:s
def read_phen_csv(fpath):
    df = read_csv_infer_if_header(fpath, nip.FORMAT_PHEN_READ_CSV)
    return df


def read_betas_csv(fpath):
    df = read_csv_infer_if_header(fpath, nip.FORMAT_BETAS_READ_CSV)
    return df


def read_scores_csv(fpath):
    df = read_csv_infer_if_header(fpath, nip.FORMAT_SCORES_READ_CSV)
    return df


def read_gwas_csv(fpath):
    df = read_csv_infer_if_header(fpath, nip.FORMAT_GWAS_READ_CSV)
    return df


def read_phen_scaling_csv(fpath):
    df = pd.read_csv(fpath, **nip.FORMAT_PHEN_SCALING_READ_CSV)
    return df


def is_continuous(data):
    '''Check if Series/np.array is contiuous by comparing to {0, 1, np.nan}'''
    if isinstance(data, pd.Series):
        data = data.unique()
    else:
        data = np.unique(data)
    is_subset = set(data).issubset({0, 1, np.nan})
    return not is_subset


# check if phen file is continuous or case control
def is_phen_df_continuous_trait(phen_df, rw=None):
    # First direct checks whether header matches
    # then look at unique entries in phen col
    if phen_df.shape[1] == len(nip.FORMAT_PHEN_CONT_COLUMNS):
        if np.all(phen_df.columns == nip.FORMAT_PHEN_CONT_COLUMNS):
            if rw:
                rw.write("trait determined continuous by column names")
            return True
        if np.all(phen_df.columns == nip.FORMAT_PHEN_CASE_COLUMNS[:-1]):
            if rw:
                rw.write("trait determined case/control by column names")
            return False
    # Check if equal length and then equal to case/control
    elif phen_df.shape[1] == len(nip.FORMAT_PHEN_CASE_COLUMNS):
        if np.all(phen_df.columns == nip.FORMAT_PHEN_CASE_COLUMNS):
            if rw:
                rw.write("trait determined case/control by column names")
            return False

    # If not determined by now, check content of phen column
    phen_set = set(phen_df.iloc[:, 3].dropna().unique())
    if rw:
        rw.write("trait determined by column content of col 4 (starting on 1)")
    if phen_set == {0, 1}:
        return False
    else:
        return True


def get_phen_col(phen_df):
    # check if continuous
    is_continuous_trait = is_phen_df_continuous_trait(phen_df)
    if is_continuous_trait:
        phen_col = "PHEN"
    else:
        phen_col = next(c for c in phen_df.columns
                        if re.search(r"CC(\.\d+)?$", c))
    return phen_col



def get_phen_cols(phen_df):
    # cols = [c for c in phen_df.columns
    #         if re.match(r"PHEN|CC(\.\d+)?", c)]
    cols = [c for c in phen_df if c not in nip.PHEN_NON_PHEN_COLS]
    return cols


def iidify(eids):
    """Return 2d numpy array [["eid1", "eid1"], ...] with eids repeated as
    strings.
    Parameters
    ----------
        eids : scalar, list, pd.Series, pd.DataFrame, pd.Index
          eids to convert.
    Returns
    -------
        np.ndarray
          2d array with reapeted string eids, like ["eid1", "eid1"].
    """
    if isinstance(eids, (pd.Series, pd.DataFrame, pd.Index)):
        eids = eids.to_numpy()
    elif isinstance(eids, str):
        eids = np.array([eids])
    elif np.isscalar(eids):
        eids = np.array([eids])
    else:
        eids = np.array(eids)
    # duplicate if not two columns
    if eids.ndim == 1:
        eids = np.array([eids, eids]).T
    elif eids.shape[1] != 2:
        eids = np.array([eids[:, 0], eids[:, 0]]).T

    eids = eids.astype(str)

    return eids


def deiidify(iids, match_return_type=None):
    """Return 1d int eids from pysnptools iids format.
    Inverse of iidify.
    Parameters
    ----------
    iids : 2d np.array
        the pysnptools preferred format [[fid, iid], ...] with the eids
        doubled.
    match_return_type : object, optional [np.array]
        return the eids with the same type as the passed object
    Returns
    np.array or list-like
        1d array of the eids in the same order as iids. Type matches
        match_return_type.
    """
    # if len(iids) == 0:
    #     # raise Exception("iids has length 0")
    #     return iids
    if match_return_type is None:
        match_return_type = np.array([])
    mrt = match_return_type
    eids = iids[:, 0].astype(int)
    if isinstance(mrt, (pd.Series, pd.DataFrame, pd.Index, list)):
        return type(mrt)(eids)
    elif isinstance(mrt, (str, int)):
        return type(mrt)(eids[0]) if len(eids) > 0 else None
    elif isinstance(mrt, np.ndarray):
        return eids
    else:
        raise Exception(f"Don't recognize return_type {type(mrt)}")


def iid_in_bed(snpreader, eids, return_mask=False):
    """Return those eids that are in snpreader.iid, or the boolean mask if
    return_mask=True.
    Parameters
    ----------
    snpreader : snpreader or snpdata (pysnptools)
        the snpreader with the bed info
    eids : list-like
        eids to subset/check
    return_mask : bool, optional [False]
        return a boolean mask for eids instead of subset
    Returns
    -------
    list-like (matching eids input type)
        the subset of eids that are in the bed matrix, or, if return_mask, a
        np.array boolean mask for eids.
    """
    iids = iidify(eids)
    mask = np.isin(iids[:, 1], snpreader.iid[:, 1])
    subset_iids = iids[mask]
    ret = deiidify(subset_iids, match_return_type=eids)
    return ret if not return_mask else mask


def iid_to_index(snpreader, iids, ignore_missing=False):
    """Return bed matrix indices for iids.
    Parameters
    ----------
    snpreader : snpreader (pysnptools)
      snpreader that will handle the mapping
    iids : list, pd.Series, pd.DataFrame, pd.Index
      eids to convert to matrix indices
    ignore_missing : bool, optional [True]
      silently ignore iids that are not in the snpreader.
    Returns
    -------
    np.ndarray
      1d array with integer indices.
    """
    # Prepare iids
    iids = iidify(iids)
    if ignore_missing:
        # iids = np.select(np.isin(iids, snpreader.iid), iids)
        iids = iids[np.isin(iids[:, 1], snpreader.iid[:, 1])]
        print(len(iids))
    return snpreader.iid_to_index(iids)


def read_bed_file(geno_basename,
                  iids,
                  top_snp_names,
                  snpreader=None,
                  is_sorting_samples=False,
                  is_sorting_snps=False,
                  read_data=True,
                  ignore_missing=False):
    """Read all relevant bed-info into memory.
    Arguments:
    geno_basename -- string with full path to bed file (with or w/o .bed)
    iids -- np.ndarray of iids only with rows (fid, iid) or list-like with eids
    top_snp_names -- list of strings with SNP-names as used by the bim-file
    Named Arguments:
    snpreader -- use this SnpReader-object instead of reading. default: None
    is_sorting_samples -- bool if sorting iids according to location in bed.
    default: False
    is_sorting_snps -- bool if sorting snps according to location in bed.
    default: False
    read_data -- bool whether to load data into memory or not. default: True
    ignore_missing -- silently ignore iids that are not in bed file.
    Returns:
    SnpData-object containing relevant data read into memory [default]
    or SnpReader-object subsetted but not read into memory if read_data=False.
    """
    if geno_basename is None:
        geno_basename = nip.GENO_BASENAME
    # read all bed-file info
    if not snpreader:
        snpreader = Bed(geno_basename, count_A1=False)
    # Sample indices
    iids = iidify(iids)
    sample_idx = iid_to_index(snpreader, iids, ignore_missing=ignore_missing)
    # SNP indx
    top_snps_idx = snpreader.sid_to_index(top_snp_names)
    # sort indices
    if is_sorting_samples:
        sample_idx.sort()
    if is_sorting_snps:
        top_snps_idx.sort()
    # subset and potentially read into memory
    sub_snpreader = snpreader[sample_idx, top_snps_idx]
    if read_data:
        snpdata = sub_snpreader.read()
        return snpdata

    return sub_snpreader


def standarize_and_nanfill_matrix(mat, means=None, stds=None):
    '''Standardize numpy matrix columnwise and replace nan with column mean.
    Arguments:
    mat -- 2D numpy array
    Named Arguments:
    means -- specified column means
    stds -- specified column stds
    Returns:
    tuple (standarized mat, means, stds)
    '''
    # Set or calc. mean for each column
    if means is not None and len(means) == len(mat):
        mat_means = means
    else:
        mat_means = np.nanmean(mat, axis=0)
    # Set or calc. std for each column
    if stds is not None and len(stds) == len(mat):
        mat_stds = stds
    else:
        mat_stds = np.nanstd(mat, axis=0)
    # Replace nans
    mat = np.nan_to_num(mat, nan=mat_means, copy=False)
    # Rescale columnwise : mat = (mat - mat_means)/mat_stds
    for col in range(mat.shape[1]):
        s = mat_stds[col]
        if s == 0.0:
            s = 1
        mat[:, col] = (mat[:, col] - mat_means[col]) / s

    return (mat, mat_means, mat_stds)


def standarize_and_nanfill_array(arr, mean=None, std=None):
    '''Standardize numpy matrix columnwise and replace nan with column mean.
    NB: edits mat inplace!
    Arguments:
    mat -- 1D numpy array
    Named Arguments:
    means -- specified column means
    stds -- specified column stds
    Returns:
    tuple (standarized mat, means, stds)
    '''
    # Set or calc. mean for each column
    if mean is not None:
        arr_mean = mean
    else:
        arr_mean = np.nanmean(arr)
    # Set or calc. std for each column
    if std is not None:
        arr_std = std
    else:
        arr_std = np.nanstd(arr)
    if arr_std == 0:
        raise Exception("arry std is zero.")
    # Replace nans
    arr = np.nan_to_num(arr, nan=arr_mean, copy=False)
    # Rescale
    arr = (arr - arr_mean) / arr_std

    return (arr, arr_mean, arr_std)


def _non_intesecting_len(sets):
    '''Take list of sets and return df with pairwise len(non-insecting
    elements).'''

    non_intersections = [[len(sr ^ sc) for sc in sets] for sr in sets]
    df = pd.DataFrame(non_intersections)
    return df


def non_intersecting_snps(x):
    '''Take list of gwas-dfs or dict with such values and return
    len(non-intersecting) of top 50k snps (according to P-value), pairwisely
    for list or for each key in dict (using union of the values).'''

    n_snps = 50000
    if type(x) == list:
        top_snp_sets = [
            set(df.sort_values("P", ascending=True)["SNP"].head(n_snps))
            for df in x
        ]
        return _non_intesecting_len(top_snp_sets)

    if type(x) == dict:
        top_snp_sets = []
        keys = list(x.keys())
        for key in keys:
            trait_top_snp_sets = [
                set(df.sort_values("P", ascending=True)["SNP"].head(n_snps))
                for df in x[key]
            ]
            trait_top_snp_set = set().union(*trait_top_snp_sets)
            top_snp_sets.append(trait_top_snp_set)
        df = _non_intesecting_len(top_snp_sets)
        df.columns = keys
        df.index = keys
        return df


def get_subG(geno_basename, iids, top_snp_names,
             snpreader=None,
             means=None, stds=None, **read_bed_kws):
    """Return (subG, geno_means, geno_stds, snpdata) for iids and snps by
    calling read_bed_file and then standardize_and_nanfill_matrix.
    Parameters
    ----------
    geno_basename : string
        full path to bed file (with or w/o .bed)
    iids : np.ndarray or list-like
        iids with rows (fid, iid) or list-like with eids
    top_snp_names : list of strings
        SNP-names as used by the bim-file
    snpreader : SnpReader, optional
        use this SnpReader-object instead of reading. default: None
    read_bed_kws : dict, optional
        keyword arguments to pass to read_bed_file
    means : list-like
        SNP mean values to use for standardization
    stds : list-like
        SNP std values to use for standardization
    Returns
    -------
    (subG, geno_means, geno_stds, snpdata)
        combined output of standardize... and read_bed_file.
    """
    snpdata = read_bed_file(geno_basename, iids, top_snp_names,
                            snpreader=snpreader,
                            **read_bed_kws,
                            read_data=True)
    subG, geno_means, geno_stds = standarize_and_nanfill_matrix(snpdata.val,
                                                                means=means,
                                                                stds=stds)
    return (subG, geno_means, geno_stds, snpdata)

