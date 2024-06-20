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


#remove train computations for much quicker performance and space saving. They are only needed for debugging issues.
            
def scoreSets(genoPATH,trait,cohortPATH,index,valsize,mlTYPE,gwasTYPE,covTYPE,trainSIZE,snpSIZE,workPATH,sex,chrm,**kwargs):
    #kwargs
    opt_params={'print_score' : 'no','l1_rat' : .5,'secret':13}
    for key, value in kwargs.items():
        if key in opt_params.keys():
            opt_params[key]=value
    l1_rat=opt_params['l1_rat']
    if mlTYPE=='ENET' or mlTYPE == 'ELOGISTIC':
        print('l1_ratio: '+str(l1_rat))
    
    #trainsizes=np.loadtxt(workPATH+'sets/train_sizes.txt')
    #m=np.max(trainsizes).astype(int)
    print(f'chromosome: {chrm}')
    m=trainSIZE
    s=snpSIZE
    print(f'train size {m}')
    print(f'snp size {s}')
    #input paths
    phenPATH = workPATH+covTYPE+".txt"
    cacoPATH = workPATH+'CACO.txt'
    # gwasPATH = workPATH+"gwas/"+gwasTYPE+"_size"+str(m)+".assoc"
    trainPATH = workPATH+"sets/train_size"+str(m)+"."+str(index)+".txt"
    valPATH = workPATH+"sets/val_size"+str(valsize)+"."+str(index)+".txt"
    
    if mlTYPE == 'LASSO' or mlTYPE == 'lasso' or mlTYPE == 'LOGISTIC':
        lp1=workPATH+"ML/"+mlTYPE+"."+covTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+".lambdas."+str(index)+".txt"
        lp2=workPATH+"ML/"+mlTYPE+"."+covTYPE+".size"+str(m)+'.chrome.'+str(chrm)+".lambdas."+str(index)+".txt"
        bp1=workPATH+"ML/"+mlTYPE+"."+covTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+".betas."+str(index)+".txt"
        bp2=workPATH+"ML/"+mlTYPE+"."+covTYPE+".size"+str(m)+'.chrome.'+str(chrm)+".betas."+str(index)+".txt"
        if os.path.isfile(lp1):
            lamPATH = lp1
            betaPATH = bp1
        else:
            lamPATH = lp1
            betaPATH = bp2
    elif mlTYPE == 'ENET' or mlTYPE == 'ELOGISTIC':
        lamPATH = workPATH+"ML/"+mlTYPE+'.l1_ratio'+str(l1_rat)+"."+covTYPE+".size"+str(m)+".lambdas."+str(index)+".txt"
        betaPATH = workPATH+"ML/"+mlTYPE+'.l1_ratio'+str(l1_rat)+"."+covTYPE+".size"+str(m)+".betas."+str(index)+".txt"
        
    if os.path.isfile(cacoPATH):
        metric="AUC"
    else:
        metric = "corr"
    
    print(workPATH)
    print(phenPATH)
    
    #output paths
    if mlTYPE == 'LASSO' or mlTYPE == 'lasso' or mlTYPE == 'LOGISTIC':
        scoreValPATH = workPATH+"scores/val."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        scoreTrainPATH = workPATH+"scores/train."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        scoreAsianPATH = workPATH+"scores/asian."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        scoreBlackPATH = workPATH+"scores/black."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        scoreChinesePATH = workPATH+"scores/chinese."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        scoreAmericanPATH = workPATH+"scores/american."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        scoreSibPATH = workPATH+"scores/sib."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
    
        metricValPATH = workPATH+"metrics/val."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        metricSibPATH = workPATH+"metrics/sib."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        metricAsianPATH = workPATH+"metrics/asian."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        metricBlackPATH = workPATH+"metrics/black."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        metricChinesePATH = workPATH+"metrics/chinese."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
        metricAmericanPATH = workPATH+"metrics/american."+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'.chrome.'+str(chrm)+"."+str(index)+".txt"
    elif mlTYPE == 'ENET' or mlTYPE == 'ELOGISTIC':
        scoreValPATH = workPATH+"scores/val."+covTYPE+".ML_"+mlTYPE+'.l1_ratio'+str(l1_rat)+".size"+str(m)+"."+str(index)+".txt"
        scoreAsianPATH = workPATH+"scores/asian."+covTYPE+".ML_"+mlTYPE+'.l1_ratio'+str(l1_rat)+".size"+str(m)+"."+str(index)+".txt"
        scoreBlackPATH = workPATH+"scores/black."+covTYPE+".ML_"+mlTYPE+'.l1_ratio'+str(l1_rat)+".size"+str(m)+"."+str(index)+".txt"
        scoreChinesePATH = workPATH+"scores/chinese."+covTYPE+".ML_"+mlTYPE+'.l1_ratio'+str(l1_rat)+".size"+str(m)+"."+str(index)+".txt"
        scoreAmericanPATH = workPATH+"scores/american."+covTYPE+".ML_"+mlTYPE+'.l1_ratio'+str(l1_rat)+".size"+str(m)+"."+str(index)+".txt"
        scoreSibPATH = workPATH+"scores/sib."+covTYPE+".ML_"+mlTYPE+'.l1_ratio'+str(l1_rat)+".size"+str(m)+"."+str(index)+".txt"
    
        metricValPATH = workPATH+"metrics/val."+covTYPE+".ML_"+mlTYPE+'.l1_ratio'+str(l1_rat)+".size"+str(m)+"."+str(index)+".txt"
        metricSibPATH = workPATH+"metrics/sib."+covTYPE+".ML_"+mlTYPE+'.l1_ratio'+str(l1_rat)+".size"+str(m)+"."+str(index)+".txt"
        metricAsianPATH = workPATH+"metrics/asian."+covTYPE+".ML_"+mlTYPE+'.l1_ratio'+str(l1_rat)+".size"+str(m)+"."+str(index)+".txt"
        metricBlackPATH = workPATH+"metrics/black."+covTYPE+".ML_"+mlTYPE+'.l1_ratio'+str(l1_rat)+".size"+str(m)+"."+str(index)+".txt"
        metricChinesePATH = workPATH+"metrics/chinese."+covTYPE+".ML_"+mlTYPE+'.l1_ratio'+str(l1_rat)+".size"+str(m)+"."+str(index)+".txt"
        metricAmericanPATH = workPATH+"metrics/american."+covTYPE+".ML_"+mlTYPE+'.l1_ratio'+str(l1_rat)+".size"+str(m)+"."+str(index)+".txt"
    
    if mlTYPE == 'LASSO' or mlTYPE == 'lasso' or mlTYPE == 'LOGISTIC':
        pruneBetaOUT = workPATH+"predictors/"+covTYPE+".ML_"+mlTYPE+".size"+str(m)+'.snps'+str(s)+'chrm'+str(chrm)+"."+str(index)+".txt"
    elif mlTYPE == 'ENET' or mlTYPE == 'ELOGISITC':
        pruneBetaOUT = workPATH+"predictors/"+covTYPE+".ML_"+mlTYPE+'.l1_ratio'+str(l1_rat)+".size"+str(m)+"."+str(index)+".txt"

    G = Bed(genoPATH,count_A1=False)
    fam = pd.read_csv(genoPATH+".fam",header=None,sep=' ')
    bim = pd.read_csv(genoPATH+".bim",header=None,sep='\t')
        
    
    phen = pd.read_csv(phenPATH,header=None,sep='\s+')
    if os.path.isfile(cacoPATH):
        caco = pd.read_csv(cacoPATH,header=None,sep='\s+')
    beta = pd.read_csv(betaPATH,header=None,sep=' ')

    # check for missing phenotypes
    #keep = ~phen.iloc[:,3].isnull()
    
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

    asianSet = pd.read_csv(cohortPATH+'asian.report.txt',header=None,sep=' ')
    blackSet = pd.read_csv(cohortPATH+'black.report.txt',header=None,sep=' ')
    chineseSet = pd.read_csv(cohortPATH+'chinese.report.txt',header=None,sep=' ')
    whiteSet = pd.read_csv(cohortPATH+'white.report.txt',header=None,sep=' ')
    americanSet = pd.read_csv(cohortPATH+'american.0.35.txt',header=None,sep=' ')

    #exclude 'american' from whiteSet
    whiteSet = whiteSet[~whiteSet[0].isin(americanSet[0])].reset_index(drop=True)

    #exclude amr used in training (other ancestry groups were excluded when making sets)
    americanSet=americanSet[~americanSet[0].isin(trainSet[0])]
    americanSet=americanSet[~americanSet[0].isin(valSet[0])]

    if sex == 'male':
        sibSet = fam[(fam[0].isin(sibSet[0]))&(fam[4]==1)][[0,1]].reset_index(drop=True)
        asianSet = fam[(fam[0].isin(asianSet[0]))&(fam[4]==1)][[0,1]].reset_index(drop=True)
        blackSet = fam[(fam[0].isin(blackSet[0]))&(fam[4]==1)][[0,1]].reset_index(drop=True)
        chineseSet = fam[(fam[0].isin(chineseSet[0]))&(fam[4]==1)][[0,1]].reset_index(drop=True)
        whiteSet = fam[(fam[0].isin(whiteSet[0]))&(fam[4]==1)][[0,1]].reset_index(drop=True)
        americanSet = fam[(fam[0].isin(americanSet[0]))&(fam[4]==1)][[0,1]].reset_index(drop=True)
    elif sex == 'female':
        sibSet = fam[(fam[0].isin(sibSet[0]))&(fam[4]==2)][[0,1]].reset_index(drop=True)
        asianSet = fam[(fam[0].isin(asianSet[0]))&(fam[4]==2)][[0,1]].reset_index(drop=True)
        blackSet = fam[(fam[0].isin(blackSet[0]))&(fam[4]==2)][[0,1]].reset_index(drop=True)
        chineseSet = fam[(fam[0].isin(chineseSet[0]))&(fam[4]==2)][[0,1]].reset_index(drop=True)
        whiteSet = fam[(fam[0].isin(whiteSet[0]))&(fam[4]==2)][[0,1]].reset_index(drop=True)
        americanSet = fam[(fam[0].isin(americanSet[0]))&(fam[4]==2)][[0,1]].reset_index(drop=True)
    
    
    #indices in fam/geno file  TIM'S SET UP HAS A PHENO FILE BASED ON 'GENOTYPED_PHENO', I.E. JUST A LIST OF PEOPLE WHO APPEAR IN BOTH PHENOTYPE FILE AND GENOTYPE FILE. THIS CAN DIFFER FROM THE FAM FILE
#    val=fam[0].astype(int).isin(valSet[0])
#    valN = pd.DataFrame(list(range(fam.shape[0])))[val].values.T[0]
    #sibs is reduced down to white sibs
#    sib=fam[0].astype(int).isin(sibSet.stack().values)
#    sibN = pd.DataFrame(list(range(fam.shape[0])))[sib].values.T[0]
#    asian=fam[0].astype(int).isin(asianSet[0])
#    asianN = pd.DataFrame(list(range(fam.shape[0])))[asian].values.T[0]
#    black=fam[0].astype(int).isin(blackSet[0])
#    blackN = pd.DataFrame(list(range(fam.shape[0])))[black].values.T[0]
#    chinese=fam[0].astype(int).isin(chineseSet[0])
#    chineseN = pd.DataFrame(list(range(fam.shape[0])))[chinese].values.T[0]

#    train = phen[0].astype(int).isin(trainSet[0])
#    trainN = pd.DataFrame(list(range(phen.shape[0])))[train].values.T[0]
    
    val=phen[0].astype(int).isin(valSet[0])
    valN = pd.DataFrame(list(range(phen.shape[0])))[val].values.T[0]
    white = phen[0].astype(int).isin(whiteSet[0].values)
    whiteN= pd.DataFrame(list(range(phen.shape[0])))[white].values.T[0]
    #sibs is reduced down to white sibs
    sib=phen[0].astype(int).isin(sibSet.stack().values)
    sibN = pd.DataFrame(list(range(phen.shape[0])))[sib & white].values.T[0]
    asian=phen[0].astype(int).isin(asianSet[0])
    asianN = pd.DataFrame(list(range(phen.shape[0])))[asian].values.T[0]
    black=phen[0].astype(int).isin(blackSet[0])
    blackN = pd.DataFrame(list(range(phen.shape[0])))[black].values.T[0]
    chinese=phen[0].astype(int).isin(chineseSet[0])
    chineseN = pd.DataFrame(list(range(phen.shape[0])))[chinese].values.T[0]
    american=phen[0].astype(int).isin(americanSet[0])
    americanN = pd.DataFrame(list(range(phen.shape[0])))[american].values.T[0]


    print("Calling into memory...",flush=True)
    # this will subset the bed matrix
    # and then actually load it into memory (.compute())
    # find missing values in the first column and count how many
    t = time.time()
    subG = G[subsetI,subsetP].read().val

#    trainG = subG[trainN,:]
    
    valG = subG[valN,:]
    sibG = subG[sibN,:]
    asianG = subG[asianN,:]
    chineseG = subG[chineseN,:]
    blackG = subG[blackN,:]
    americanG = subG[americanN,:]
    subG = 0.0
    elapsed = time.time() - t
    print(elapsed)
    # note, 1000 columns loads basically as quick as 1
    print("Final shapes: val, sib, asian, chinese, black, american")
    print(valG.shape)
    print(sibG.shape)
    print(asianG.shape)
    print(chineseG.shape)
    print(blackG.shape)
    print(americanG.shape)

    print("Calc means")
    # calculate column means with no missing values
    # nanmean calculates mean skipping nan
    center = beta[6]
    
    print("NA repl")
    # na replacement
#    missing = np.argwhere(np.isnan(trainG))
#    for row in range(0,missing.shape[0]):
#        ind1 = missing[row,0]
#        ind2 = missing[row,1]
#        trainG[ind1,ind2] = center[ind2]
        
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

    missing = np.argwhere(np.isnan(asianG))
    for row in range(0,missing.shape[0]):
        ind1 = missing[row,0]
        ind2 = missing[row,1]
        asianG[ind1,ind2] = center[ind2]

    missing = np.argwhere(np.isnan(chineseG))
    for row in range(0,missing.shape[0]):
        ind1 = missing[row,0]
        ind2 = missing[row,1]
        chineseG[ind1,ind2] = center[ind2]

    missing = np.argwhere(np.isnan(blackG))
    for row in range(0,missing.shape[0]):
        ind1 = missing[row,0]
        ind2 = missing[row,1]
        blackG[ind1,ind2] = center[ind2]

    missing = np.argwhere(np.isnan(americanG))
    for row in range(0,missing.shape[0]):
        ind1 = missing[row,0]
        ind2 = missing[row,1]
        americanG[ind1,ind2] = center[ind2]

        
    print("Center")
    for col in range(0,valG.shape[1]):
#        trainG[:,col] = trainG[:,col] - center[col]
        valG[:,col] = valG[:,col] - center[col]
        sibG[:,col] = sibG[:,col] - center[col]
        asianG[:,col] = asianG[:,col] - center[col]
        chineseG[:,col] = chineseG[:,col] - center[col]
        blackG[:,col] = blackG[:,col] - center[col]
        americanG[:,col] = americanG[:,col] - center[col]

#    trainScore = np.dot(trainG,(beta.iloc[:,7:beta.shape[1]]))
#    trainPhen = phen.iloc[trainN,2]

    valScore = np.dot(valG,(beta.iloc[:,7:beta.shape[1]]))
    valPhen = phen.iloc[valN,2]
    sibScore = np.dot(sibG,(beta.iloc[:,7:beta.shape[1]]))
    sibPhen = phen.iloc[sibN,2]
    asianScore = np.dot(asianG,(beta.iloc[:,7:beta.shape[1]]))
    asianPhen = phen.iloc[asianN,2]
    chineseScore = np.dot(chineseG,(beta.iloc[:,7:beta.shape[1]]))
    chinesePhen = phen.iloc[chineseN,2]
    blackScore = np.dot(blackG,(beta.iloc[:,7:beta.shape[1]]))
    blackPhen = phen.iloc[blackN,2]
    americanScore = np.dot(americanG,(beta.iloc[:,7:beta.shape[1]]))
    americanPhen = phen.iloc[americanN,2]

    
    if os.path.isfile(cacoPATH):
        valCaco = caco.iloc[valN,2]
        blackCaco = caco.iloc[blackN,2]
        sibCaco = caco.iloc[sibN,2]
        asianCaco = caco.iloc[asianN,2]
        chineseCaco = caco.iloc[chineseN,2]
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
        metricAsian = np.array([skl.metrics.roc_auc_score(asianCaco, asianScore[:,best])])
        metricBlack = np.array([skl.metrics.roc_auc_score(blackCaco, blackScore[:,best])])
        metricChinese = np.array([skl.metrics.roc_auc_score(chineseCaco, chineseScore[:,best])])
        metricAmerican = np.array([skl.metrics.roc_auc_score(americanCaco, americanScore[:,best])])
    else:
        metricSib = np.array([pd.concat((pd.DataFrame(sibPhen).reset_index(drop=True),pd.DataFrame(sibScore[:,best]).reset_index(drop=True)),axis=1).corr().iloc[0,1]])
        metricAsian = np.array([pd.concat((pd.DataFrame(asianPhen).reset_index(drop=True),pd.DataFrame(asianScore[:,best]).reset_index(drop=True)),axis=1).corr().iloc[0,1]])
        metricBlack = np.array([pd.concat((pd.DataFrame(blackPhen).reset_index(drop=True),pd.DataFrame(blackScore[:,best]).reset_index(drop=True)),axis=1).corr().iloc[0,1]])
        metricChinese = np.array([pd.concat((pd.DataFrame(chinesePhen).reset_index(drop=True),pd.DataFrame(chineseScore[:,best]).reset_index(drop=True)),axis=1).corr().iloc[0,1]])
        metricAmerican = np.array([pd.concat((pd.DataFrame(americanPhen).reset_index(drop=True),pd.DataFrame(americanScore[:,best]).reset_index(drop=True)),axis=1).corr().iloc[0,1]])
    
    pd.DataFrame(metricSib).to_csv(metricSibPATH,header=False,index=False)
    pd.DataFrame(metricAsian).to_csv(metricAsianPATH,header=False,index=False)
    pd.DataFrame(metricBlack).to_csv(metricBlackPATH,header=False,index=False)
    pd.DataFrame(metricChinese).to_csv(metricChinesePATH,header=False,index=False)
    pd.DataFrame(metricAmerican).to_csv(metricAmericanPATH,header=False,index=False)

    print("Val "+str(metric)+" ("+str(trait)+"): "+str(metricVal[best]))
    print("Sibling (white) "+str(metric)+": "+str(metricSib))
    print("Asian "+str(metric)+": "+str(metricAsian))
    print("Black "+str(metric)+": "+str(metricBlack))
    print("Chinese "+str(metric)+": "+str(metricChinese))
    print("American "+str(metric)+": "+str(metricAmerican))

    keepcols = [0,1,2,3,4,5,6,7+best]
    outbeta = beta.iloc[:,keepcols]
    outbeta.to_csv(r''+pruneBetaOUT,sep=' ',index=False,header=False)
    
    if opt_params['print_score'] == 'yes':
        #toprint_valscore=pd.concat((phen.iloc[valPhen.index][1].reset_index(drop=True),pd.DataFrame(valScore[:,best])),axis=1).rename(columns={1:"IID",0:"SCORE"})
        #toprint_trainscore=pd.concat((phen.iloc[trainPhen.index][1].reset_index(drop=True),pd.DataFrame(trainScore)),axis=1)
        
        toprint_valscore=pd.concat((phen.iloc[valPhen.index][1].reset_index(drop=True),pd.DataFrame(valScore)),axis=1)
        toprint_sibscore=pd.concat((phen.iloc[sibPhen.index][1].reset_index(drop=True),pd.DataFrame(sibScore[:,best])),axis=1).rename(columns={1:"IID",0:"SCORE"})
        toprint_asianscore=pd.concat((phen.iloc[asianPhen.index][1].reset_index(drop=True),pd.DataFrame(asianScore[:,best])),axis=1).rename(columns={1:"IID",0:"SCORE"})
        toprint_blackscore=pd.concat((phen.iloc[blackPhen.index][1].reset_index(drop=True),pd.DataFrame(blackScore[:,best])),axis=1).rename(columns={1:"IID",0:"SCORE"})
        toprint_chinesescore=pd.concat((phen.iloc[chinesePhen.index][1].reset_index(drop=True),pd.DataFrame(chineseScore[:,best])),axis=1).rename(columns={1:"IID",0:"SCORE"})
        toprint_americanscore=pd.concat((phen.iloc[americanPhen.index][1].reset_index(drop=True),pd.DataFrame(americanScore[:,best])),axis=1).rename(columns={1:"IID",0:"SCORE"})

        #toprint_trainscore.to_csv(scoreTrainPATH,header=True,index=False)
        
        toprint_valscore.to_csv(scoreValPATH,header=True,index=False)
        toprint_sibscore.to_csv(scoreSibPATH,header=True,index=False)
        toprint_asianscore.to_csv(scoreAsianPATH,header=True,index=False)
        toprint_blackscore.to_csv(scoreBlackPATH,header=True,index=False)
        toprint_chinesescore.to_csv(scoreChinesePATH,header=True,index=False)
        toprint_americanscore.to_csv(scoreAmericanPATH,header=True,index=False)
        
        
######################
######################
#TEMPORARY OUTPUT HERE
######################
######################
#     trainPATH = workPATH+"sets/train_size"+str(415035)+"."+str(index)+".txt"
#     trainSet = pd.read_csv(trainPATH,header=None,sep=' ')
#     train=phen[0].astype(int).isin(trainSet[0])
#     trainN = pd.DataFrame(list(range(phen.shape[0])))[train].values.T[0]
#     trainPhen = phen.iloc[trainN,2]
#     mu = np.mean(trainPhen)
#     sig = np.std(trainPhen)
#     tmpmetric1 = np.array([skl.metrics.r2_score((valPhen-mu)/sig, valScore[:,i]) for i in range(0,valScore.shape[1])])
#     adj=np.array((len(beta))/(len(beta)-np.array([len(beta[beta[7+i]!=0]) for i in range(0,valScore.shape[1])])))
#     tmpmetric2 = 1-(1-np.array(tmpmetric1))*adj
#     newScore=np.zeros(np.shape(valScore))
#     for i in range(0,valScore.shape[1]):
#         if i != 0:    
#             newScore[:,i]=valScore[:,i]/np.std(valScore[:,i])
#     tmpmetric3 = np.array([skl.metrics.r2_score((valPhen-mu)/sig, newScore[:,i]) for i in range(0,valScore.shape[1])])
#     tmpmetric4 = 1-(1-np.array(tmpmetric3))*adj
#     np.savetxt('/mnt/home/rabentim/tmp1.txt',tmpmetric1)
#     np.savetxt('/mnt/home/rabentim/tmp2.txt',tmpmetric2)
#     np.savetxt('/mnt/home/rabentim/tmp3.txt',tmpmetric3)
#     np.savetxt('/mnt/home/rabentim/tmp4.txt',tmpmetric4)
#######################
#######################
#TEMPORARY OUTPUT ABOVE
#######################
#######################
        
        
        
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
