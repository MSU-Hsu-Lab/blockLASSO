import time as time
import tracemalloc
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.linear_model import lasso_path
from pysnptools.snpreader import Bed


def global_vs_block_lasso():
    work_path = '/mnt/home/rabentim/simulations'
    gene_path = f'{work_path}/simulated' # '/full/path/to/plink/files'
    bim = pd.read_csv(f'{gene_path}.bim',header=None,sep='\t')
    fam = pd.read_csv(f'{gene_path}.fam',header=None,sep='\s+')

    #error relative size
    err_rel_size = 5
    data = []
    for nsnps_per in [10,50,100,500]:
        for nsample in [5000,10000,15000,20000]:
            for s in [0.1,0.5,0.9]:
    # for nsnps_per in [10]:
    #     for nsample in [5000]:
    #         for s in [0.1]:

                #specify the number of blocks (in this case the number of chromosomes)
                nblocks=22
            
                #list of snps to use. nsnps_per are randomly sampled from each block. This must be edited if you want to work on non-chromosome blocks
                all_snps = [bim[bim[0]==i][1][0:nsnps_per] for i in np.sort(np.random.choice(range(1,23),nblocks,replace=False))]
                #fam file of just samples
                sub_fam = fam.sample(nsample)
                sub_fam = sub_fam.sort_values(by=1)
                #snpreader Bed file
                bed_data = Bed(gene_path,count_A1=False)
                print(f'using {nsample} samples and {nsnps_per*nblocks} features')
    
    
                tracemalloc.start()
                tracemalloc.reset_peak()
                t = time.time()
                iids = sub_fam[[0,1]].values.astype(str)
                sample_idx = bed_data.iid_to_index(iids)
                top_snps_idx = bed_data.sid_to_index(np.hstack(all_snps))
                sub_snpreader = bed_data[sample_idx, top_snps_idx]
                snpdata = sub_snpreader.read()
                subG=snpdata.val
                ellapse = time.time() - t
                print(f'{ellapse} seconds')
                malloc = tracemalloc.get_traced_memory()
                malloc = tuple(it/1000000000 for it in malloc)
                print(f'tracemalloc: {malloc} GB of RAM')
                print(f'final shape: {np.shape(subG)}')
                global_malloc = malloc
                tracemalloc.stop()
    
                #separate genotype matrix into training, validation, and testing sets: train 80%, val 10%, test 10% 
                trainG = subG[0:np.floor(.8*nsample).astype(int)]
                valG = subG[np.floor(.8*nsample).astype(int):np.floor(.9*nsample).astype(int)]
                testG = subG[np.floor(.9*nsample).astype(int):]
    
                #standardize the genetic matrices according to the training set
                center = np.zeros(trainG.shape[1])
                spread = np.zeros(trainG.shape[1])
                for col in range(0,subG.shape[1]):
                    center[col] = np.nanmean(trainG[:,col])
                    spread[col] = np.nanstd(trainG[:,col])
                
                missing = np.argwhere(np.isnan(subG))
                for row in range(0,missing.shape[0]):
                    ind1 = missing[row,0]
                    ind2 = missing[row,1]
                    subG[ind1,ind2] = center[ind2]
                
                for col in range(0,subG.shape[1]):
                    val = spread[col]
                    if spread[col] == 0.0:
                        val = 1.0
                    subG[:,col] = (subG[:,col] - center[col])/val
                
                trainG = subG[0:np.floor(.8*nsample).astype(int)]
                valG = subG[np.floor(.8*nsample).astype(int):np.floor(.9*nsample).astype(int)]
                testG = subG[np.floor(.9*nsample).astype(int):]
    
    
                #define number of participating features and assign them effect weights in [0,1). 
                # err_rel_size defines the error scale in terms of the standard deviation of the true weight effects
                
                ntrue_weights = np.floor(s*nsnps_per).astype(int)
                tmp = np.zeros(nblocks*nsnps_per)
                weights = np.random.rand(ntrue_weights)
                weight_locations = np.sort(np.random.randint(0,nblocks*nsnps_per,ntrue_weights))
                tmp[weight_locations] = weights
                weights = tmp
                gene_effects = np.dot(subG,weights)
                error_effects = err_rel_size*np.std(gene_effects)* np.random.rand(nsample)
                
                #define true phenotype and split into various sets
                true_y = gene_effects + error_effects
                train_y = true_y[0:np.floor(.8*nsample).astype(int)]
                val_y = true_y[np.floor(.8*nsample).astype(int):np.floor(.9*nsample).astype(int)]
                test_y = true_y[np.floor(.9*nsample).astype(int):]
                
                #standardize training set for algorithm training
                ymu = np.mean(train_y)
                ysig = np.std(train_y)
                norm_train_y = (train_y-ymu)/ysig



                #run global lasso
                tracemalloc.start()
                tracemalloc.reset_peak()
                t = time.time()
                path = skl.linear_model.lasso_path(trainG,norm_train_y,n_alphas=50,eps=.01,max_iter=1500)
                ellapse = time.time() - t
                print(f'{ellapse} seconds')
                malloc = tracemalloc.get_traced_memory()
                malloc = tuple(it/1000000000 for it in malloc)
                print(f'tracemalloc: {malloc} GB of RAM')
                global_lasso_malloc = malloc
                tracemalloc.stop()


                # correlation in validation set
                val_mets =np.array([np.corrcoef(val_y,np.dot(valG,path[1])[:,i])[0][1] for i in range(0,50)])
                max_ind = np.nanargmax(val_mets)
                # correlation in testing set
                global_met = np.corrcoef(test_y,np.dot(testG,path[1][:,max_ind]))[0][1]
                #True correlation
                true_met = np.corrcoef(gene_effects,true_y)[0][1]


                #move on to block version
                del subG; del trainG; del valG; del testG

                tracemalloc.start()
                tracemalloc.reset_peak()
                t = time.time()
                iids = sub_fam[[0,1]].values.astype(str)
                sample_idx = bed_data.iid_to_index(iids)
                for i in range(0,nblocks):
                    top_snps_idx = bed_data.sid_to_index(all_snps[i])
                    sub_snpreader = bed_data[sample_idx, top_snps_idx]
                    snpdata = sub_snpreader.read()
                    tmpG=snpdata.val
                    if i ==0:
                        malloc = tracemalloc.get_traced_memory()
                        malloc = tuple(it/1000000000 for it in malloc)
                        print(f'single (first) block tracemalloc: {malloc} GB of RAM')
                        single_block_malloc = malloc
                
                        subG=tmpG
                    else:
                        subG = np.hstack((subG,tmpG))
                ellapse = time.time() - t
                print(f'{ellapse} seconds')
                malloc = tracemalloc.get_traced_memory()
                malloc = tuple(it/1000000000 for it in malloc)
                print(f'total tracemalloc: {malloc} GB of RAM')
                print(f'final shape: {np.shape(subG)}')
                block_malloc = malloc
                tracemalloc.stop()

                # train 80%, val 10%, test 10% 
                trainG = subG[0:np.floor(.8*nsample).astype(int)]
                valG = subG[np.floor(.8*nsample).astype(int):np.floor(.9*nsample).astype(int)]
                testG = subG[np.floor(.9*nsample).astype(int):]

                center = np.zeros(trainG.shape[1])
                spread = np.zeros(trainG.shape[1])
                for col in range(0,subG.shape[1]):
                    center[col] = np.nanmean(trainG[:,col])
                    spread[col] = np.nanstd(trainG[:,col])
                
                missing = np.argwhere(np.isnan(subG))
                for row in range(0,missing.shape[0]):
                    ind1 = missing[row,0]
                    ind2 = missing[row,1]
                    subG[ind1,ind2] = center[ind2]
                
                for col in range(0,subG.shape[1]):
                    val = spread[col]
                    if spread[col] == 0.0:
                        val = 1.0
                    subG[:,col] = (subG[:,col] - center[col])/val
                
                trainG = subG[0:np.floor(.8*nsample).astype(int)]
                valG = subG[np.floor(.8*nsample).astype(int):np.floor(.9*nsample).astype(int)]
                testG = subG[np.floor(.9*nsample).astype(int):]


                paths = []
                block_lasso_mallocs = []
                tracemalloc.start()
                tracemalloc.reset_peak()
                t = time.time()
                for i in range(0,nblocks):
                    tmp_path = skl.linear_model.lasso_path(trainG[:,i*nsnps_per:(i+1)*nsnps_per],
                                                   norm_train_y,n_alphas=50,eps=.01,max_iter=1500)
                    paths.append(tmp_path)
                    ellapse = time.time() - t
                    print(f'{ellapse} seconds')
                    malloc = tracemalloc.get_traced_memory()
                    block_lasso_mallocs.append(malloc)
                    malloc = tuple(it/1000000000 for it in malloc)
                    print(f'tracemalloc: {malloc} GB of RAM')
                tracemalloc.stop()

                #build validation correlations for each block
                valGs =[]
                for i in range(0,nblocks):
                    valGs.append(valG[:,i*nsnps_per:(i+1)*nsnps_per])
                vals_mets = np.array([np.array([np.corrcoef(val_y,np.dot(valGs[j],paths[j][1])[:,i])[0][1] for i in range(0,50)]) for j in range(0,nblocks)])
                bests = np.array([np.nanargmax(vals_mets[i]) for i in range(0,nblocks)])
                
                #perform linear regression on blocks
                block_val_scores = np.array([np.dot(valGs[i],paths[i][1][:,bests[i]]) for i in range(0,nblocks)])
                fit = skl.linear_model.LinearRegression().fit(block_val_scores.T,val_y)
                
                #apply full model to testing set
                testGs =[]
                for i in range(0,nblocks):
                    testGs.append(testG[:,i*nsnps_per:(i+1)*nsnps_per])
                block_test_scores = np.array([np.dot(testGs[i],paths[i][1][:,bests[i]]) for i in range(0,nblocks)])
                
                unweighted_met = np.corrcoef(np.sum(block_test_scores.T,axis=1),test_y)[0][1]
                block_met = np.corrcoef(fit.predict(block_test_scores.T),test_y)[0][1]


                
                

                data.append([nsnps_per,nsample,s,nblocks,
                             *global_malloc,
                             *global_lasso_malloc,
                             true_met,global_met,
                            *single_block_malloc,
                            *block_malloc,
                            *(np.mean(np.array(block_lasso_mallocs),axis=0)/1000000000),
                            unweighted_met,block_met])
    data = pd.DataFrame(data,columns=['nsnps_per','nsample','sparsity','nblocks',
                                      'global_malloc_current','global_malloc_peak',
                                      'global_lasso_malloc_current','global_lasso_malloc_peak',
                                      'true_met','global_met',
                                      'single_block_malloc_current','single_block_malloc_peak',
                                      'block_malloc_current','block_malloc_peak',
                                     'avg_block_lasso_malloc_current','avg_block_lasso_malloc_peak',
                                     'unweighted_block_met','block_met'])
    data.to_csv(f'{work_path}/lasso_grid_data.txt',header=True,index=False)

    return 0

def main():
    print('running')
    global_vs_block_lasso()
    print('done running')

if __name__ == "__main__":
    main()

