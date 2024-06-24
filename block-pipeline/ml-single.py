import numpy as np
import pandas as pd
import scipy as sp
import math as math
import time as time
import argparse
import textwrap
import os
from sklearn import linear_model
import sklearn as skl
from pysnptools.snpreader import Bed
import names_in_pipeline as nip
import pipeline_utilities as pu
import multiprocessing as mp
 
def runML(genPATH,trait,index,gwasTYPE,covTYPE,MLTYPE,workPATH,trainSIZE,snpSIZE,chrm,plinktype,**kwargs):
    print(trait)
    print(workPATH)
    print('cv fold: '+str(index))
    #get training size
    m=trainSIZE
    print('train size: '+str(m))
    # constants
    nstep = 100
    print(str(nstep)+' steps in ML path')
    lamratio = 0.0005
    if gwasTYPE != 'CACO':
        lamratio = 0.001
    print('lambda ratio: '+str(lamratio))
    chrm = int(chrm)
    
    #kwargs
    opt_params={'l1_rat' : .5, 'secret':13}
    for key, value in kwargs.items():
        if key in opt_params.keys():
            opt_params[key]=value
    l1_rat=opt_params['l1_rat']
    if MLTYPE=='ENET' or MLTYPE=='ELOGISTIC':
        print('l1_ratio: '+str(l1_rat))
    
    if MLTYPE == 'LOGISTIC' or MLTYPE == 'ELOGISTIC':
        n_alphas = 100
    
    print('load paths')
    #input paths
    if gwasTYPE =='CACO':
        if plinktype == 1:
            gwasPATH = f'{workPATH}gwas/{gwasTYPE}_size{m}.{index}.assoc'
        else:
            gwasPATH = f'{workPATH}gwas/{gwasTYPE}_size{m}.{index}.PHENO1.glm.logistic.hybrid'
    else:
        gwasPATH = f'{workPATH}gwas/{gwasTYPE}_size{m}.{index}.PHENO1.glm.linear'

    phenPATH = workPATH+covTYPE+'.txt'
    trainPATH = workPATH+'sets/train_size'+str(m)+'.'+str(index)+'.txt'
    # output paths
    if MLTYPE == 'LASSO':
        lamPATH = workPATH+"ML/"+MLTYPE+"."+covTYPE+".size"+str(m)+'.snps'+str(snpSIZE)+'.chrome.'+str(chrm)+".lambdas."+str(index)+".txt"
        betaPATH = workPATH+"ML/"+MLTYPE+"."+covTYPE+".size"+str(m)+'.snps'+str(snpSIZE)+'.chrome.'+str(chrm)+".betas."+str(index)+".txt"
        gapPATH = workPATH+"ML/"+MLTYPE+"."+covTYPE+".size"+str(m)+'.snps'+str(snpSIZE)+'.chrome.'+str(chrm)+".duality-gap."+str(index)+".txt"
     
    print('load fam/bim/phen/gwas')    
    G = Bed(genPATH,count_A1=False)
    fam = pd.read_csv(genPATH+".fam",header=None,sep=' ')
    bim = pd.read_csv(genPATH+".bim",header=None,sep='\t')

    phen = pd.read_csv(phenPATH,header=None,sep='\s+',names=['FID','IID','PHENO'])
    # blank,chr, snp, bp, a1, fa, fu, a2 ,x2, P, OR, blank
    gwas = pd.read_csv(gwasPATH,sep='\s+',dtype={'#CHROM':'str'})

    #block size
    top = snpSIZE
    print(f'SNP per chrom: {top}')
    print('compute subsets')
    # sort gwas into top N snps

    # excluding the sex chromosomes (and MT)
    if plinktype ==1:
    # the following two lines are for plink 1.9
        sexchr = bim[0].astype(int).eq(chrm)
        best=gwas[sexchr].sort_values(by='P',ascending=True)['SNP'][0:top]
    else:
        #the following is for plink2.0, it also now includes an explicit MAF filter!
        best=gwas[(gwas['#CHROM']==f'{chrm}')&(gwas['A1_FREQ']>=0.001)].sort_values(by='P',ascending=True)['ID'][0:top]

    
    subsetP = bim[1].isin(best)
    subsetP = np.stack(pd.DataFrame(list(range(bim.shape[0])))[subsetP].values,axis=1)[0]

    #load training indeces
    train = np.loadtxt(trainPATH,dtype=int)
    train_inds = phen['IID'].isin(train.T[0])
    
    print('load BED data')
    bed_data = Bed(genPATH,count_A1=False)

    snpdata = pu.read_bed_file("",
                               phen['IID'],
                               best,
                               snpreader=bed_data,
                               is_sorting_samples=False,
                               is_sorting_snps=True,
                               read_data=True)


    subG=snpdata.val[train_inds]
    target_phen = phen['PHENO'].loc[train_inds].values

                       
    print("Calc means")
    # calculate column means with no missing values
    # nanmean calculates mean skipping nan
    center = np.zeros(subG.shape[1])
    spread = np.zeros(subG.shape[1])
    for col in range(0,subG.shape[1]):
        center[col] = np.nanmean(subG[:,col])
        spread[col] = np.nanstd(subG[:,col])

    print("NA repl")     
    # na replacement
    missing = np.argwhere(np.isnan(subG))
    for row in range(0,missing.shape[0]):
        ind1 = missing[row,0]
        ind2 = missing[row,1]
        subG[ind1,ind2] = center[ind2]

    print("Standardize")    
    # standardize the columns
    for col in range(0,subG.shape[1]):
        val = spread[col]
        if spread[col] == 0.0:
            val = 1.0
        subG[:,col] = (subG[:,col] - center[col])/val

    y = target_phen
    # standardize the phenotype
    if MLTYPE == 'LASSO' or MLTYPE == 'ENET':
        ymu = np.mean(y)
        ysig = np.std(y)
        y = (y-ymu)/ysig
        
    # do the lasso
    print("Begin "+str(MLTYPE),flush=True)    
    t = time.time()
    if MLTYPE == 'LASSO':
        path = skl.linear_model.lasso_path(subG,y,n_alphas=nstep,eps=lamratio,n_iter=1500)

    elapsed = time.time() - t
    print(str(MLTYPE)+" time:",flush=True)
    print(elapsed)
    
    if MLTYPE == 'LASSO': 
        betas = path[1]
        lamb = path[0]
        gap = path[2]
        gap = pd.DataFrame(gap)
        gap.to_csv(r''+gapPATH,sep=' ',index=False,header=False)
        
    metadat = bim.iloc[subsetP,:]
    metadat = metadat.reset_index(drop=True)
    if MLTYPE == 'LASSO' or MLTYPE == 'ENET':
        betas = pd.DataFrame(np.transpose(np.transpose(betas)*np.transpose(ysig/spread)))
    else:
        betas = pd.DataFrame(betas)
    lamb = pd.DataFrame(lamb)

    out = pd.concat([metadat,pd.DataFrame(center),betas],ignore_index=True,axis=1)
    out.to_csv(r''+betaPATH,sep = ' ',index=False,header=False)
    lamb.to_csv(r''+lamPATH,sep=' ',index=False,header=False)


    return 0





def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     prog='ML',
                                     usage='%(ML)s',
                                     description='''Runs the ML path algo for lasso, enet, and l1-logistic.''')

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

    required_named.add_argument('--cv-fold',
                                type=str,
                                required=True,
                                help='index variable, 1-5')
    
    required_named.add_argument('--gwas-type',
                                type=str,
                                required=True,
                                help='start of gwas file name')
    
    required_named.add_argument('--cov-type',
                                type=str,
                                required=True,
                                help='start of pheno file name to regress on')
    required_named.add_argument('--ml-type',
                                type=str,
                                required=True,
                                help='start of pheno file name to regress on')

    # file to
    required_named.add_argument('--working-path',
                                type=str,
                                required=True,
                                help='Where all the output goes')
    # train size
    required_named.add_argument('--train-size',
                                type=int,
                                required=True,
                                help='training size')
    # train size
    required_named.add_argument('--snp-size',
                                type=int,
                                required=True,
                                help='snp set size')
    # chromosome
    required_named.add_argument('--chrm',
                                type=int,
                                required=True,
                                help='chromosome')
    # plink type
    required_named.add_argument('--plinktype',
                                type=int,
                                required=True,
                                help='version of plink')

    # optional arguments
    optional_named = parser.add_argument_group('Optional named arguments')
    optional_named.add_argument('--l1-ratio',
                                type=float,
                                required=False,
                                help='l1 ratio for enet')
    
    args = parser.parse_args()
    
    
    if args.l1_ratio is not None:
        runML(args.geno_path,args.trait,args.cv_fold,args.gwas_type,args.cov_type,args.ml_type,args.working_path,args.train_size,args.snp_size,args.chrm,l1_rat=args.l1_ratio)
    else:
        runML(args.geno_path,args.trait,args.cv_fold,args.gwas_type,args.cov_type,args.ml_type,args.working_path,args.train_size,args.snp_size,args.chrm,args.plinktype)


exit(main())
