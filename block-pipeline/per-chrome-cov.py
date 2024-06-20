import numpy as np
import pandas as pd
import scipy as sp
import math as math
import time as time
import argparse
import textwrap
import os
import hsupkg as hp
#from pandas_plink import read_plink
from sklearn import linear_model
import sklearn as skl
from pysnptools.snpreader import Bed
from ukbphenreader import PhenReader
pr = PhenReader()
            
def covariance(genoPATH,trait,cohortPATH,index,ssize,covTYPE,workPATH,sex,**kwargs):
    #kwargs
    opt_params={'secret':13}
    for key, value in kwargs.items():
        if key in opt_params.keys():
            opt_params[key]=value
    workPATH = f'/mnt/home/rabentim/methods/{trait}'
    sibs = pd.read_csv(f'{cohortPATH}/sibPairs.txt',header=None,sep='\s+')
    fam = pd.read_csv(f'{genoPATH}.fam',header=None,sep='\s+')
    trainsizes = pd.read_csv(f'{workPATH}/sets/train_sizes.txt',header=None)
    snp_sizes=pd.read_csv(f'{workPATH}/sets/snp_sizes.txt',header=None)

    siblist = pd.DataFrame(pd.concat((sibs[0],sibs[1]))).drop_duplicates().reset_index(drop=True)
    tmp = siblist.set_index(0)
    pr.assign_ancestry(tmp)
    sibs = tmp[tmp['ANCESTRY']=='white']
    sibs = sibs[sibs.index.isin(fam[0])]
    sibs = sibs.sort_index()
    pr.assign_sex(sibs)
    if sex =='male':
        sibs = sibs[sibs['SEX']==1]
    elif sex == 'female':
        sibs = sibs[sibs['SEX']==0]
        
    m = trainsizes.max().values[0]
    s = int(ssize)
    cv = index

    # for c in range(1,23):
    #     val = pd.read_csv(f'{workPATH}/metrics/val.{covTYPE}.ML_LASSO.size{m}.snps{s}.chrome.{c}.{cv}.txt',header=None,sep='\s+')
    #     lasso = pd.read_csv(f'{workPATH}/ML/LASSO.{covTYPE}.size{m}.snps{s}.chrome.{c}.betas.{cv}.txt',header=None,sep='\s+')
            
    #     max_ind = val.idxmax().values[0]
    #     if c ==1:
    #         pred = lasso[lasso[7+max_ind] != 0][[0,1,2,3,4,5,6,7+max_ind]]
    coefs = np.load(f'{workPATH}/coefs.npy')
    if len(coefs)==1:
        sind = 0
    else:
        sind = np.where(snp_sizes.values.T[0]==s)[0][0]
    for c in range(1,23):
        val = pd.read_csv(f'{workPATH}/metrics/val.{covTYPE}.ML_LASSO.size{m}.snps{s}.chrome.{c}.{cv}.txt',header=None,sep='\s+')
        lasso = pd.read_csv(f'{workPATH}/ML/LASSO.{covTYPE}.size{m}.snps{s}.chrome.{c}.betas.{cv}.txt',header=None,sep='\s+')
            
        max_ind = val.idxmax().values[0]
        tmp = lasso[lasso[7+max_ind] != 0][[0,1,2,3,4,5,6,7+max_ind]]
        tmp = tmp.rename(columns={(7+max_ind):'beta'})
        tmp['raw_beta'] = tmp['beta']
        tmp['beta'] = tmp['beta']*coefs[sind][int(cv)-1][c-1]
        if c ==1:
            pred = tmp
        else:
            pred = pd.concat([pred,tmp]).reset_index(drop=True)
        
    snps = pred[1]
    mafs = pd.read_csv(f'{workPATH}/sets/maf.size{m}.{cv}.frq',sep='\s+')

    print('loading genotype data')
    tmp = hp.get_genodata(iids=sibs.index, snps = snps.values, read_data = True).val

    #NAN replacement
    for i in range(0,np.shape(tmp)[0]):
        for j in range(0,np.shape(tmp)[1]):
            if np.isnan(tmp[i,j]):
                tmp[i,j] = pred[6].values[j]
    ctmp = tmp-np.tile(pred[6].values,(len(sibs),1))
    xbeta = ctmp.T*(pred['beta'].values[:,None])
    xbeta = pd.DataFrame(xbeta.T)

    cov = xbeta.cov()

    raw_xbeta = ctmp.T*(pred['raw_beta'].values[:,None])
    raw_xbeta = pd.DataFrame(raw_xbeta.T)

    raw_cov = raw_xbeta.cov()
    
    f=mafs[mafs['SNP'].isin(pred[1])]['MAF'].values
    ssv = 2*pred['beta'].values**2*f*(1-f)



    clengths = np.array([249250621, 243199373, 198022430, 191154276, 180915260, 
            171115067, 159138663, 146364022, 141213431, 135534747,
            135006516, 133851895, 115169878, 107349540, 102531392,
            90354753, 81195210, 78077248, 59128983, 63025520, 48129895, 51304566])
    ctots=np.array([np.sum(clengths[0:i]) for i in range(0,len(clengths)+1)])
    totpos=[]
    for i in range(0,len(pred)):
        totpos.append(ctots[pred[0].iloc[i]-1]+pred[3].iloc[i])
    totpos = np.array(totpos)

    pred.loc[:,'new var'] = (cov.sum()/cov.sum().sum()).values
    pred.loc[:,'new raw var'] = (raw_cov.sum()/raw_cov.sum().sum()).values
    pred.loc[:,'totpos']=totpos
    pred.loc[:,'old ssv']=ssv
    pred.loc[:,'new ssv'] = np.diag(cov)/np.diag(cov).sum()

    pred.to_csv(f'{workPATH}/predictors/pred.size{m}.snps{s}.{cv}.txt')
    
    return 0

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     prog='scoreSets',
                                     usage='%(scoreSets)s',
                                     description='''Choosing lambda and scores.''')

    # essential arguments
    required_named = parser.add_argument_group('Required named arguments')
    required_named.add_argument('--geno-path',
                                type=str,
                                required=True,
                                help='path to genotypes')

    required_named.add_argument('--trait',
                                type=str,
                                required=True,
                                help='name of trait')

    required_named.add_argument('--cohort-path',
                                type=str,
                                required=True,
                                help='path to cohorts')

    required_named.add_argument('--array-id',
                                type=str,
                                required=True,
                                help='CV fold')
    
    required_named.add_argument('--snp-size',
                                type=str,
                                required=True,
                                help='N snps per chrome')

    required_named.add_argument('--cov-type',
                                type=str,
                                required=True,
                                help='type of coveriates adjusted for')
    # train size
    required_named.add_argument('--sex',
                                type=str,
                                required=True,
                                help='sex: male, female, or both')

    


    # file to
    required_named.add_argument('--working-directory',
                                type=str,
                                required=True,
                                help='Where all the output goes')
    

    args = parser.parse_args()
    covariance(args.geno_path, args.trait, args.cohort_path, args.array_id,
                  args.snp_size, args.cov_type, args.working_directory, args.sex)


exit(main())
