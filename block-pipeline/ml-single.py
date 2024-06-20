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
    #trainsizes=np.loadtxt(workPATH+'sets/train_sizes.txt')
    #m=np.max(trainsizes).astype(int)
    m=trainSIZE
    print('train size: '+str(m))
    # constants
    #nstep = 190
    nstep = 100
    print(str(nstep)+' steps in ML path')
    # origininally lamratio = 0.04
    # lamratio of 0.01 gets hgt predictors to ~25k snps
    # setting it to 0.04 again...
    # block lasso i usually set lamratio = 0.005. testing smaller lamratios
    lamratio = 0.0005
    if trait == 'bioMarkers2.19':
        lamratio = 0.004
    if gwasTYPE != 'CACO':
        lamratio = 0.001
    if trait == 'Lipoprotein.A':
        lamratio = 0.0004
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
    if MLTYPE =='LOGISTIC' or MLTYPE =='ELOGISTIC':
        phenPATH = workPATH+'CACO.txt'
    else:
        phenPATH = workPATH+covTYPE+'.txt'
    trainPATH = workPATH+'sets/train_size'+str(m)+'.'+str(index)+'.txt'
    # output paths
    if MLTYPE == 'LASSO':
        lamPATH = workPATH+"ML/"+MLTYPE+"."+covTYPE+".size"+str(m)+'.snps'+str(snpSIZE)+'.chrome.'+str(chrm)+".lambdas."+str(index)+".txt"
        betaPATH = workPATH+"ML/"+MLTYPE+"."+covTYPE+".size"+str(m)+'.snps'+str(snpSIZE)+'.chrome.'+str(chrm)+".betas."+str(index)+".txt"
        gapPATH = workPATH+"ML/"+MLTYPE+"."+covTYPE+".size"+str(m)+'.snps'+str(snpSIZE)+'.chrome.'+str(chrm)+".duality-gap."+str(index)+".txt"
    elif MLTYPE == 'ENET':
        lamPATH = workPATH+"ML/"+MLTYPE+'.l1_ratio'+str(l1_rat)+"."+covTYPE+".size"+str(m)+".lambdas."+str(index)+".txt"
        betaPATH = workPATH+"ML/"+MLTYPE+'.l1_ratio'+str(l1_rat)+"."+covTYPE+".size"+str(m)+".betas."+str(index)+".txt"
        gapPATH = workPATH+"ML/"+MLTYPE+'.l1_ratio'+str(l1_rat)+"."+covTYPE+".size"+str(m)+".duality-gap."+str(index)+".txt"
    elif MLTYPE == 'LOGISTIC':
        lamPATH = workPATH+"ML/"+MLTYPE+"."+covTYPE+".size"+str(m)+".lambdas."+str(index)+".txt"
        betaPATH = workPATH+"ML/"+MLTYPE+"."+covTYPE+".size"+str(m)+".betas."+str(index)+".txt"
        gapPATH = workPATH+"ML/"+MLTYPE+"."+covTYPE+".size"+str(m)+".duality-gap."+str(index)+".txt"
    elif MLTYPE == 'ELOGISTIC':
        lamPATH = workPATH+"ML/"+MLTYPE+'.l1_ratio'+str(l1_rat)+"."+covTYPE+".size"+str(m)+".lambdas."+str(index)+".txt"
        betaPATH = workPATH+"ML/"+MLTYPE+'.l1_ratio'+str(l1_rat)+"."+covTYPE+".size"+str(m)+".betas."+str(index)+".txt"
        gapPATH = workPATH+"ML/"+MLTYPE+'.l1_ratio'+str(l1_rat)+"."+covTYPE+".size"+str(m)+".duality-gap."+str(index)+".txt"
    print('load fam/bim/phen/gwas')    
    #    (bim,fam,G) = read_plink(genPATH)
    #    fam = pd.DataFrame(fam.values)
    #    bim = pd.DataFrame(bim.values)
    G = Bed(genPATH,count_A1=False)
    fam = pd.read_csv(genPATH+".fam",header=None,sep=' ')
    bim = pd.read_csv(genPATH+".bim",header=None,sep='\t')

    phen = pd.read_csv(phenPATH,header=None,sep='\s+',names=['FID','IID','PHENO'])
    # blank,chr, snp, bp, a1, fa, fu, a2 ,x2, P, OR, blank
    gwas = pd.read_csv(gwasPATH,sep='\s+',dtype={'#CHROM':'str'})
    
    #Number of SNPs used, standard is 50k
#     if MLTYPE == 'LOGISTIC' or MLTYPE == 'ELOGISTIC':
#         top = 10000 #50000
#     else:
#         top = snpSIZE
#     print('using '+str(top)+' SNPs')
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

    #load training indeces  BE CAREFUL IF LOADING A JULIA TRAIN SET
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
    elif MLTYPE == 'ENET':
        path = skl.linear_model.enet_path(subG,y,l1_ratio=l1_rat,n_alphas=nstep,eps=lamratio,n_iter=1500)
    elif MLTYPE == 'LOGISTIC':
        #n_path=100
        #lamb=np.logspace(-4,0,n_path)
        Xy=np.dot(subG.T,y)
        Xy=Xy[:,np.newaxis]
        alpha_max=np.sqrt(np.sum(Xy**2,axis=1)).max()/(np.shape(subG)[0])#*l1_rat)
        alpha_min = 1/(np.shape(subG)[0]*alpha_max)
        #lamb=np.logspace(np.log10(alpha_max/100), np.log10(alpha_max/2), num=n_alphas)[::-1]    #also might try just using alpha_max as top point
        lamb=np.logspace(np.log10(alpha_min), np.log10(alpha_min*10), num=n_alphas)[::-1]
#        print('penalization array: ', lamb)
        betas=np.zeros((top,len(lamb)))
        intercept = np.zeros(len(lamb))
        tol = 2e-2
        print('Logisit tolerance is: %f' % tol)
        global t_log_fit
        def t_log_fit(c, lamb, subG, y): #, betas, intercept): final inputs for version 3
            penal_log = skl.linear_model.LogisticRegression(C=c,penalty='l1',tol=tol,solver='saga',max_iter=3000,multi_class='auto',n_jobs=-1) #defaul tol=1e-4
            path = penal_log.fit(subG,y)
            #for version 3
            #betas[:,np.where(lamb==c)[0][0]] = path.coef_
            #intercept[np.where(lamb==c)[0][0]] = path.intercept_
            return c, path.coef_, path.intercept_
#version 1
        #pool = mp.Pool(mp.cpu_count())
        #for c in lamb:
        #    tmp = pool.apply_async(t_log_fit,args=(c, lamb, subG, y)) #for c in lamb
        #    betas[:,np.where(lamb==c)[0][0]] = tmp.get()[1]
        #    intercept[np.where(lamb==c)[0][0]] = tmp.get()[2]
        #pool.close()
        #pool.join()

#Version 2  THIS SEEMS TO BE THE VERSION THAT WORKS!!!!!!
        print('Number of CPUs in use: ', mp.cpu_count())
        pool = mp.Pool(mp.cpu_count()-1,maxtasksperchild=1000)
        fits = [pool.apply_async(t_log_fit,args=(c, lamb, subG, y)) for c in lamb]
        pool.close()
        pool.join()
        for i in range(0,len(lamb)):
            betas[:,np.where(lamb==fits[i].get()[0])[0][0]] = fits[i].get()[1]
            intercept[np.where(lamb==fits[i].get()[0])[0][0]] = fits[i].get()[2]
            
#Version 3
        #pool = mp.Pool(mp.cpu_count())
        #[pool.apply_async(t_log_fit,args=(c, lamb, penal_log, z_matrix,wp_df, betas, intercept)) for c in lamb]
        #pool.close()
        #pool.join
#old version
#        for c in lamb: 
#            penal_log=skl.linear_model.LogisticRegression(C=c,penalty='l1',tol=1e-4,solver='saga',multi_class='auto',n_jobs=-1)
#            path = penal_log.fit(subG,y)
#            betas[:,np.where(lamb==c)[0][0]] = path.coef_
#            intercept[np.where(lamb==c)[0][0]] = path.intercept_
    elif MLTYPE == 'ELOGISTIC':
        n_path=100
        lamb=np.logspace(-4,0,n_path)
        betas=np.zeros((top,len(lamb)))
        intercept = np.zeros(len(lamb))
        for c in lamb:
            path = skl.linear_model.LogisticRegression(C=c,penalty='elasticnet',tol=1e-4,solver='saga',l1_ratio=l1_rat,multi_class='auto',n_jobs=-1).fit(subG,y)
            betas[:,np.where(lamb==c)[0][0]] = path.coef_
            intercept[np.where(lamb==c)[0][0]] = path.intercept_
    elapsed = time.time() - t
    print(str(MLTYPE)+" time:",flush=True)
    print(elapsed)
    
    if MLTYPE == 'LASSO' or MLTYPE == 'ENET': #only the lasso and enet get a gap
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
