library(bigsnpr)
options(bigstatsr.check.parallel.blas = FALSE)

out1 = 'hgt_test.multicor_20.cv2.allsnp.betaFinal.txt'
out2 = 'hgt_test.multicor_20.cv2.allsnp.intercept.txt'
cross_valSets = 2

NCORE=nb_cores()

print("VARY MULTICORE")
print("START")
print(Sys.time())

print("CORES")
print(NCORE)

print("CROSS VAL SETS")
print(cross_valSets)

bedpath="/mnt/home/lellolou/hsuGroup/ukb500/genotypes/calls.merged/ukb500.calls.gpsnp.bed"
rdspath = "/mnt/home/lellolou/hsuGroup/ukb500/genotypes/calls.merged/ukb500.calls.gpsnp.rds"
obj.bigSNP <- snp_attach(rdspath)

print("Parse big SNP data...")

G0   <- obj.bigSNP$genotypes
CHR <- obj.bigSNP$map$chromosome
POS <- obj.bigSNP$map$physical.pos
ID  <- obj.bigSNP$map$marker.ID

print("initial parsing done.")
print("start missing value impute")

G <- snp_fastImputeSimple(G0, ncores = nb_cores())
obj.bigSNP$genotypes <- G

print("Done. now read phenotypes")

# now read in phenotype
phenpath = "/mnt/home/lellolou/siblingGP/hgt.whiteRep/hgt.whiteRep.pruned.txt"
trainpath = "/mnt/home/lellolou/siblingGP/hgt.whiteRep/TrainSet.1.txt"

phen = read.delim(phenpath, sep=" ")
train = read.delim(trainpath, sep=" ", header=FALSE) # WARNING ZERO INDEXED
subsetN = train[,1]+1

print("Subset phen and genotype")
y <- phen[subsetN,4]

# writing bed to attach
print("writing bed")
outpath = '/mnt/home/lellolou/scratch/bigstatsr/cleaned.bed'
outbed <- snp_subset(obj.bigSNP, ind.row = subsetN)

print('attaching rds')
rds <- snp_attach(outbed)
X = snp_attach(outbed)

print("dimension G")
print(dim(X$genotypes))

# ready to train
print('training')
mod <- big_spLinReg(X$genotypes, y, K = cross_valSets, alphas=c(1), ncores=nb_cores())

outbeta = X$map
outbeta['beta'] = summary(mod)$beta
int = summary(mod)$intercept

write.table(outbeta, out1, sep=' ', row.names=FALSE, col.names=TRUE, quote=FALSE)
write.table(int, out2, sep=' ', row.names=FALSE, col.names=TRUE, quote=FALSE)

print("END")
print(Sys.time())