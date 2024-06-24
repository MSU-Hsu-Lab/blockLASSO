import numpy as np
import pandas as pd
import scipy as sp
import math as math
import time as time
import argparse
import textwrap
import os
#from pandas_plink import read_plink
from sklearn import linear_model
import sklearn as skl
from pysnptools.snpreader import Bed

            
def scoreSets(genoPATH,trait,cohortPATH,index,valsize,mlTYPE,gwasTYPE,covTYPE,trainSIZE,snpSIZE,workPATH,sex,chrm,**kwargs):
    #kwargs
    opt_params={'print_score' : 'no','l1_rat' : .5,'secret':13}
    for key, value in kwargs.items():
        if key in opt_params.keys():
            opt_params[key]=value
    l1_rat=opt_params['l1_rat']
    
    print(f'chromosome: {chrm}')
    m=trainSIZE
    s=snpSIZE
    print(f'train size {m}')
    print(f'snp size {s}')
    #input paths
    phenPATH = workPATH+covTYPE+".txt"
    cacoPATH = workPATH+'CACO.txt'
    trainPATH = workPATH+"sets/train_size"+str(m)+"."+str(index)+".txt"
    valPATH = workPATH+"sets/val_size"+str(valsize)+"."+str(index)+".txt"
    
    if mlTYPE == 'LASSO':
        lamPATH=workPATH+"ML/"+mlTYPE+"."+covTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+".lambdas."+str(index)+".txt"
        betaPATH=workPATH+"ML/"+mlTYPE+"."+covTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+".betas."+str(index)+".txt"

    #check if a 'WORKPATH/CACO.txt' file exists to check if it's a binary trait
    if os.path.isfile(cacoPATH):
        metric="AUC"
    else:
        metric = "corr"
    
    print(workPATH)
    print(phenPATH)
    
    #output paths
    if mlTYPE == 'LASSO':
        scoreValPATH = workPATH+"scores/val."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        scoreTrainPATH = workPATH+"scores/train."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        scoresasPATH = workPATH+"scores/sas."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        scoreafrPATH = workPATH+"scores/afr."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        scoreeasPATH = workPATH+"scores/eas."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        scoreAmericanPATH = workPATH+"scores/american."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        scoreSibPATH = workPATH+"scores/sib."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
    
        metricValPATH = workPATH+"metrics/val."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        metricSibPATH = workPATH+"metrics/sib."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        metricsasPATH = workPATH+"metrics/sas."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        metricafrPATH = workPATH+"metrics/afr."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        metriceasPATH = workPATH+"metrics/eas."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        metricAmericanPATH = workPATH+"metrics/american."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
    
    if mlTYPE == 'LASSO':
        pruneBetaOUT = workPATH+"predictors/"+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'chrm'+str(chrm)+"."+str(index)+".txt"

    G = Bed(genoPATH,count_A1=False)
    fam = pd.read_csv(genoPATH+".fam",header=None,sep=' ')
    bim = pd.read_csv(genoPATH+".bim",header=None,sep='\t')
        
    
    phen = pd.read_csv(phenPATH,header=None,sep='\s+')
    if os.path.isfile(cacoPATH):
        caco = pd.read_csv(cacoPATH,header=None,sep='\s+')
    beta = pd.read_csv(betaPATH,header=None,sep=' ')

    
    names = beta[1]
    subsetP = bim[1].isin(names)
    subsetP = np.stack(pd.DataFrame(list(range(bim.shape[0])))[subsetP].values,axis=1)[0]
    iids = phen[1]
    subsetI = fam[0].isin(iids)
    subsetI = np.stack(pd.DataFrame(list(range(fam.shape[0])))[subsetI].values,axis=1)[0]
    
    print("read sets...")
    
    trainSet = pd.read_csv(trainPATH,header=None,sep=' ') 
    valSet = pd.read_csv(valPATH,header=None,sep=' ')

    sibSet = pd.read_csv(cohortPATH+'sibPairs.txt',header=None,sep=' ')
    #reduce sibs down to unique individuals
    sibSet=pd.DataFrame(pd.concat((sibSet[0],sibSet[1]),ignore_index=True))
    sibSet = sibSet.sort_values([0]).groupby(0).first().reset_index()

    sasSet = pd.read_csv(cohortPATH+'sas.report.txt',header=None,sep=' ')
    afrSet = pd.read_csv(cohortPATH+'afr.report.txt',header=None,sep=' ')
    easSet = pd.read_csv(cohortPATH+'eas.report.txt',header=None,sep=' ')
    eurSet = pd.read_csv(cohortPATH+'eur.report.txt',header=None,sep=' ')
    americanSet = pd.read_csv(cohortPATH+'american.0.35.txt',header=None,sep=' ')

    #exclude 'american' from EUR cohort (possible overlaps)
    eurSet = eurSet[~eurSet[0].isin(americanSet[0])].reset_index(drop=True)

    #exclude amr used in training (other ancestry groups were excluded when making sets)
    americanSet=americanSet[~americanSet[0].isin(trainSet[0])]
    americanSet=americanSet[~americanSet[0].isin(valSet[0])]

    if sex == 'male':
        sibSet = fam[(fam[0].isin(sibSet[0]))&(fam[4]==1)][[0,1]].reset_index(drop=True)
        sasSet = fam[(fam[0].isin(sasSet[0]))&(fam[4]==1)][[0,1]].reset_index(drop=True)
        afrSet = fam[(fam[0].isin(afrSet[0]))&(fam[4]==1)][[0,1]].reset_index(drop=True)
        easSet = fam[(fam[0].isin(easSet[0]))&(fam[4]==1)][[0,1]].reset_index(drop=True)
        eurSet = fam[(fam[0].isin(eurSet[0]))&(fam[4]==1)][[0,1]].reset_index(drop=True)
        americanSet = fam[(fam[0].isin(americanSet[0]))&(fam[4]==1)][[0,1]].reset_index(drop=True)
    elif sex == 'female':
        sibSet = fam[(fam[0].isin(sibSet[0]))&(fam[4]==2)][[0,1]].reset_index(drop=True)
        sasSet = fam[(fam[0].isin(sasSet[0]))&(fam[4]==2)][[0,1]].reset_index(drop=True)
        afrSet = fam[(fam[0].isin(afrSet[0]))&(fam[4]==2)][[0,1]].reset_index(drop=True)
        easSet = fam[(fam[0].isin(easSet[0]))&(fam[4]==2)][[0,1]].reset_index(drop=True)
        eurSet = fam[(fam[0].isin(eurSet[0]))&(fam[4]==2)][[0,1]].reset_index(drop=True)
        americanSet = fam[(fam[0].isin(americanSet[0]))&(fam[4]==2)][[0,1]].reset_index(drop=True)
    
    
    val=phen[0].astype(int).isin(valSet[0])
    valN = pd.DataFrame(list(range(phen.shape[0])))[val].values.T[0]
    eur = phen[0].astype(int).isin(eurSet[0].values)
    eurN= pd.DataFrame(list(range(phen.shape[0])))[eur].values.T[0]
    #sibs is reduced down to EUR sibs
    sib=phen[0].astype(int).isin(sibSet.stack().values)
    sibN = pd.DataFrame(list(range(phen.shape[0])))[sib & eur].values.T[0]
    sas=phen[0].astype(int).isin(sasSet[0])
    sasN = pd.DataFrame(list(range(phen.shape[0])))[sas].values.T[0]
    afr=phen[0].astype(int).isin(afrSet[0])
    afrN = pd.DataFrame(list(range(phen.shape[0])))[afr].values.T[0]
    eas=phen[0].astype(int).isin(easSet[0])
    easN = pd.DataFrame(list(range(phen.shape[0])))[eas].values.T[0]
    american=phen[0].astype(int).isin(americanSet[0])
    americanN = pd.DataFrame(list(range(phen.shape[0])))[american].values.T[0]


    print("Calling into memory...",flush=True)
    # this will subset the bed matrix
    # and then actually load it into memory (.compute())
    # find missing values in the first column and count how many
    t = time.time()
    subG = G[subsetI,subsetP].read().val
    
    valG = subG[valN,:]
    sibG = subG[sibN,:]
    sasG = subG[sasN,:]
    easG = subG[easN,:]
    afrG = subG[afrN,:]
    americanG = subG[americanN,:]
    subG = 0.0
    elapsed = time.time() - t
    print(elapsed)
    # note, 1000 columns loads basically as quick as 1
    print("Final shapes: val, sib, sas, eas, afr, american")
    print(valG.shape)
    print(sibG.shape)
    print(sasG.shape)
    print(easG.shape)
    print(afrG.shape)
    print(americanG.shape)

    print("Calc means")
    # calculate column means with no missing values (can be loaded from ML file)
    center = beta[6]
    
    print("NA repl")
    # na replacement  
    missing = np.argwhere(np.isnan(valG))
    for row in range(0,missing.shape[0]):
        ind1 = missing[row,0]
        ind2 = missing[row,1]
        valG[ind1,ind2] = center[ind2]
        
    missing = np.argwhere(np.isnan(sibG))
    for row in range(0,missing.shape[0]):
        ind1 = missing[row,0]
        ind2 = missing[row,1]
        sibG[ind1,ind2] = center[ind2]

    missing = np.argwhere(np.isnan(sasG))
    for row in range(0,missing.shape[0]):
        ind1 = missing[row,0]
        ind2 = missing[row,1]
        sasG[ind1,ind2] = center[ind2]

    missing = np.argwhere(np.isnan(easG))
    for row in range(0,missing.shape[0]):
        ind1 = missing[row,0]
        ind2 = missing[row,1]
        easG[ind1,ind2] = center[ind2]

    missing = np.argwhere(np.isnan(afrG))
    for row in range(0,missing.shape[0]):
        ind1 = missing[row,0]
        ind2 = missing[row,1]
        afrG[ind1,ind2] = center[ind2]

    missing = np.argwhere(np.isnan(americanG))
    for row in range(0,missing.shape[0]):
        ind1 = missing[row,0]
        ind2 = missing[row,1]
        americanG[ind1,ind2] = center[ind2]

        
    print("Center")
    for col in range(0,valG.shape[1]):
        valG[:,col] = valG[:,col] - center[col]
        sibG[:,col] = sibG[:,col] - center[col]
        sasG[:,col] = sasG[:,col] - center[col]
        easG[:,col] = easG[:,col] - center[col]
        afrG[:,col] = afrG[:,col] - center[col]
        americanG[:,col] = americanG[:,col] - center[col]

    valScore = np.dot(valG,(beta.iloc[:,7:beta.shape[1]]))
    valPhen = phen.iloc[valN,2]
    sibScore = np.dot(sibG,(beta.iloc[:,7:beta.shape[1]]))
    sibPhen = phen.iloc[sibN,2]
    sasScore = np.dot(sasG,(beta.iloc[:,7:beta.shape[1]]))
    sasPhen = phen.iloc[sasN,2]
    easScore = np.dot(easG,(beta.iloc[:,7:beta.shape[1]]))
    easPhen = phen.iloc[easN,2]
    afrScore = np.dot(afrG,(beta.iloc[:,7:beta.shape[1]]))
    afrPhen = phen.iloc[afrN,2]
    americanScore = np.dot(americanG,(beta.iloc[:,7:beta.shape[1]]))
    americanPhen = phen.iloc[americanN,2]

    
    if os.path.isfile(cacoPATH):
        valCaco = caco.iloc[valN,2]
        afrCaco = caco.iloc[afrN,2]
        sibCaco = caco.iloc[sibN,2]
        sasCaco = caco.iloc[sasN,2]
        easCaco = caco.iloc[easN,2]
        americanCaco = caco.iloc[americanN,2]
    
    metricVal = np.zeros(valScore.shape[1])

    print("SIZE VALSCORE:")
    print(valScore.shape)
    
    for k in range(0,valScore.shape[1]):
        if os.path.isfile(cacoPATH):
            metricVal[k] = skl.metrics.roc_auc_score(valCaco, valScore[:,k])
        else:
            metricVal[k] = np.corrcoef(valPhen,valScore[:,k])[0,1]

    pd.DataFrame(metricVal).to_csv(metricValPATH,header=False,index=False)

    
    best = np.nanargmax(metricVal)
    print('best val metric at index: '+str(best))
    if os.path.isfile(cacoPATH):
        metricSib = np.array([skl.metrics.roc_auc_score(sibCaco, sibScore[:,best])])
        metricsas = np.array([skl.metrics.roc_auc_score(sasCaco, sasScore[:,best])])
        metricafr = np.array([skl.metrics.roc_auc_score(afrCaco, afrScore[:,best])])
        metriceas = np.array([skl.metrics.roc_auc_score(easCaco, easScore[:,best])])
        metricAmerican = np.array([skl.metrics.roc_auc_score(americanCaco, americanScore[:,best])])
    else:
        metricSib = np.array([pd.concat((pd.DataFrame(sibPhen).reset_index(drop=True),pd.DataFrame(sibScore[:,best]).reset_index(drop=True)),axis=1).corr().iloc[0,1]])
        metricsas = np.array([pd.concat((pd.DataFrame(sasPhen).reset_index(drop=True),pd.DataFrame(sasScore[:,best]).reset_index(drop=True)),axis=1).corr().iloc[0,1]])
        metricafr = np.array([pd.concat((pd.DataFrame(afrPhen).reset_index(drop=True),pd.DataFrame(afrScore[:,best]).reset_index(drop=True)),axis=1).corr().iloc[0,1]])
        metriceas = np.array([pd.concat((pd.DataFrame(easPhen).reset_index(drop=True),pd.DataFrame(easScore[:,best]).reset_index(drop=True)),axis=1).corr().iloc[0,1]])
        metricAmerican = np.array([pd.concat((pd.DataFrame(americanPhen).reset_index(drop=True),pd.DataFrame(americanScore[:,best]).reset_index(drop=True)),axis=1).corr().iloc[0,1]])
    
    pd.DataFrame(metricSib).to_csv(metricSibPATH,header=False,index=False)
    pd.DataFrame(metricsas).to_csv(metricsasPATH,header=False,index=False)
    pd.DataFrame(metricafr).to_csv(metricafrPATH,header=False,index=False)
    pd.DataFrame(metriceas).to_csv(metriceasPATH,header=False,index=False)
    pd.DataFrame(metricAmerican).to_csv(metricAmericanPATH,header=False,index=False)

    print("Val "+str(metric)+" ("+str(trait)+"): "+str(metricVal[best]))
    print("Sibling (eur) "+str(metric)+": "+str(metricSib))
    print("sas "+str(metric)+": "+str(metricsas))
    print("afr "+str(metric)+": "+str(metricafr))
    print("eas "+str(metric)+": "+str(metriceas))
    print("American "+str(metric)+": "+str(metricAmerican))

    keepcols = [0,1,2,3,4,5,6,7+best]
    outbeta = beta.iloc[:,keepcols]
    outbeta.to_csv(r''+pruneBetaOUT,sep=' ',index=False,header=False)
    
    if opt_params['print_score'] == 'yes':
        toprint_valscore=pd.concat((phen.iloc[valPhen.index][1].reset_index(drop=True),pd.DataFrame(valScore)),axis=1)
        toprint_sibscore=pd.concat((phen.iloc[sibPhen.index][1].reset_index(drop=True),pd.DataFrame(sibScore[:,best])),axis=1).rename(columns={1:"IID",0:"SCORE"})
        toprint_sasscore=pd.concat((phen.iloc[sasPhen.index][1].reset_index(drop=True),pd.DataFrame(sasScore[:,best])),axis=1).rename(columns={1:"IID",0:"SCORE"})
        toprint_afrscore=pd.concat((phen.iloc[afrPhen.index][1].reset_index(drop=True),pd.DataFrame(afrScore[:,best])),axis=1).rename(columns={1:"IID",0:"SCORE"})
        toprint_easscore=pd.concat((phen.iloc[easPhen.index][1].reset_index(drop=True),pd.DataFrame(easScore[:,best])),axis=1).rename(columns={1:"IID",0:"SCORE"})
        toprint_americanscore=pd.concat((phen.iloc[americanPhen.index][1].reset_index(drop=True),pd.DataFrame(americanScore[:,best])),axis=1).rename(columns={1:"IID",0:"SCORE"})
        
        toprint_valscore.to_csv(scoreValPATH,header=True,index=False)
        toprint_sibscore.to_csv(scoreSibPATH,header=True,index=False)
        toprint_sasscore.to_csv(scoresasPATH,header=True,index=False)
        toprint_afrscore.to_csv(scoreafrPATH,header=True,index=False)
        toprint_easscore.to_csv(scoreeasPATH,header=True,index=False)
        toprint_americanscore.to_csv(scoreAmericanPATH,header=True,index=False)
        
        
        
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
    required_named.add_argument('--val-size',
                                type=str,
                                required=True,
                                help='size of validation set')
    required_named.add_argument('--ml-type',
                                type=str,
                                required=True,
                                help='type of ML algorithm used')
    required_named.add_argument('--gwas-type',
                                type=str,
                                required=True,
                                help='type of phenotype used for gwas SNP selection')
    required_named.add_argument('--cov-type',
                                type=str,
                                required=True,
                                help='type of coveriates adjusted for')
    # train size
    required_named.add_argument('--train-size',
                                type=int,
                                required=True,
                                help='training size')
    # train size
    required_named.add_argument('--sex',
                                type=str,
                                required=True,
                                help='sex: male, female, or both')
    # top SNP size
    required_named.add_argument('--snp-size',
                                type=str,
                                required=True,
                                help='number of top SNPs used')
    # chromosome
    required_named.add_argument('--chrm',
                                type=int,
                                required=True,
                                help='chromosome')

    
    # optional arguments
    optional_named = parser.add_argument_group('Optional named arguments')
    optional_named.add_argument('--l1-ratio',
                                type=float,
                                required=False,
                                help='l1 ratio for enet')
    optional_named.add_argument('--print-score',
                                type=str,
                                required=False,
                                help='Do you want to print scores?')

    # file to
    required_named.add_argument('--working-directory',
                                type=str,
                                required=True,
                                help='Where all the output goes')
    

    args = parser.parse_args()
    if args.print_score is None:
        args.print_score='no'
    if args.l1_ratio is not None:
        scoreSets(args.geno_path,args.trait,args.cohort_path,args.array_id,
                  args.val_size,args.ml_type,args.gwas_type,args.cov_type,args.train_size,args.snp_size,
                  args.working_directory,args.sex,args.chrm,l1_rat=args.l1_ratio,print_score=args.print_score)
    else:
        scoreSets(args.geno_path,args.trait,args.cohort_path,args.array_id,
                  args.val_size,args.ml_type,args.gwas_type,args.cov_type,args.train_size,args.snp_size,
                  args.working_directory,args.sex,args.chrm,print_score=args.print_score)

exit(main())
