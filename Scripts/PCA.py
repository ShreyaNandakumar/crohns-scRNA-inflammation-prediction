import numpy as np

# ── file paths ─────────────────────────────────────────────────────────────────
# these are the sample-level files created by the preprocessing pipeline
# X = gene expression matrix (49 samples x 2000 genes)
# y = labels (1 = inflamed, 0 = non-inflamed)
# groups = patient IDs (used to make sure we split by patient, not randomly)
xPath      = "/Users/shreyabalamurugan/Machine Learning for Scientists/FINAL PROJECT/sample_level_X.npy"
yPath      = "/Users/shreyabalamurugan/Machine Learning for Scientists/FINAL PROJECT/sample_level_y.npy"
groupsPath = "/Users/shreyabalamurugan/Machine Learning for Scientists/FINAL PROJECT/sample_level_groups.npy"

# this is where we will save the PCA results for kNN and XGBoost to use
outputPath = "Results/PCA/pca_splits.npy"


# ── load the data ──────────────────────────────────────────────────────────────
# np.load reads a .npy file back into a numpy array
X      = np.load(xPath)
y      = np.load(yPath)
groups = np.load(groupsPath, allow_pickle=True)  # allow_pickle=True needed for string arrays

# print a summary so we can verify the data loaded correctly
print(f"X shape: {X.shape}")  # should be (49, 2000)
print(f"Inflamed samples: {np.sum(y==1)}, Non-inflamed samples: {np.sum(y==0)}")
print(f"Number of unique patients: {len(np.unique(groups))}")  # should be 16


# ── PCA helper functions ───────────────────────────────────────────────────────

def pcaFit(xTrain, nComponents):
    """
    Fit PCA on the training data only.

    PCA finds the directions (principal components) in the data that
    capture the most variation. We do this using SVD (Singular Value
    Decomposition), which is a standard matrix math operation.

    Important: we only use the TRAINING data to fit PCA.
    We never let the test data influence the components.

    Parameters:
        xTrain     : the training gene expression matrix (nSamples x nGenes)
        nComponents: how many principal components to keep (e.g. 5, 10, 15)

    Returns:
        components            : the principal component directions (nComponents x nGenes)
        mean                  : the mean of each gene across training samples
        explainedVarianceRatio: fraction of TOTAL variance each PC captures
    """

    # step 1: compute the mean of each gene across all training samples
    # this gives us a vector of length nGenes
    mean = xTrain.mean(axis=0)

    # step 2: subtract the mean from every sample (centering the data)
    # centering is required before SVD so that PCA captures variance, not mean shifts
    xCentered = xTrain - mean

    # step 3: run SVD on the centered training matrix
    # SVD decomposes xCentered into three matrices: U, S, Vt
    # U     = information about the samples
    # S     = singular values (how much variance each component explains)
    # Vt    = the principal component directions (what we care about)
    # full_matrices=False gives us a compact version which is faster
    U, S, Vt = np.linalg.svd(xCentered, full_matrices=False)

    # step 4: keep only the top nComponents rows of Vt
    # each row is one principal component direction
    components = Vt[:nComponents]  # shape: (nComponents, nGenes)

    # step 5: compute how much variance each kept PC explains
    # variance = singular value squared / (nSamples - 1)
    explainedVariance = (S[:nComponents] ** 2) / (xTrain.shape[0] - 1)

    # step 6: compute total variance across ALL components (not just the ones we kept)
    # this is the correct denominator so the ratio reflects the full dataset variance
    # without this step the ratio would always sum to 100% which is misleading
    totalVariance = (S ** 2) / (xTrain.shape[0] - 1)

    # step 7: divide each component's variance by the TOTAL variance across all components
    # now the ratio tells us what fraction of the full dataset variance each PC captures
    # e.g. 0.35 means that PC explains 35% of all variation in the 2000 genes
    explainedVarianceRatio = explainedVariance / totalVariance.sum()

    return components, mean, explainedVarianceRatio


def pcaTransform(X, components, mean):
    """
    Project data into PCA space using components and mean from training data.

    This works for BOTH train and test data.
    The key rule: always use the training mean to center — never the test mean.
    Using the test mean would be data leakage.

    Parameters:
        X         : the data to transform (nSamples x nGenes)
        components: the principal component directions from pcaFit()
        mean      : the training mean from pcaFit()

    Returns:
        xPca: the projected data (nSamples x nComponents)
    """

    # center the data using the TRAINING mean (not this data's own mean)
    xCentered = X - mean

    # project onto the principal components
    # matrix multiply: (nSamples x nGenes) @ (nGenes x nComponents)
    # result shape: (nSamples x nComponents)
    xPca = xCentered @ components.T

    return xPca


# ── grouped cross-validation splitter ─────────────────────────────────────────

def groupedKFoldSplits(groups, nFolds=5):
    """
    Split the data into folds such that all samples from the same patient
    always stay together — they never appear in both train and test.

    This is called grouped cross-validation. It is required here because
    we have multiple biopsies per patient, and if the same patient appeared
    in both train and test, the model would be tested on data it essentially
    already saw, which would inflate our results.

    Parameters:
        groups: array of patient IDs, one per sample (length 49)
        nFolds: how many folds to create (default 5)

    Returns:
        splits: list of (trainIndices, testIndices) tuples, one per fold
    """

    # get the list of unique patient IDs
    uniquePatients = np.unique(groups)  # e.g. ['CD1', 'CD2', ..., 'CD16']

    # assign each patient to a fold number using modulo
    # e.g. patient 0 -> fold 0, patient 1 -> fold 1, ..., patient 5 -> fold 0 again
    patientFoldMap = {
        patient: index % nFolds
        for index, patient in enumerate(uniquePatients)
    }

    # build the actual train/test index splits for each fold
    splits = []

    for foldNumber in range(nFolds):

        # test = samples whose patient is assigned to this fold
        testMask = np.array([patientFoldMap[g] == foldNumber for g in groups])

        # train = everything else
        trainMask = ~testMask

        # convert boolean masks to actual integer indices
        trainIndices = np.where(trainMask)[0]
        testIndices  = np.where(testMask)[0]

        splits.append((trainIndices, testIndices))

    return splits


# ── run PCA inside each CV fold ────────────────────────────────────────────────

# we will try three different numbers of principal components
# this lets us later compare which number works best for kNN and XGBoost
nComponentsList = [5, 10, 15]

# number of cross-validation folds
nFolds = 5

print(f"\nRunning PCA inside {nFolds}-fold grouped cross-validation")
print(f"Trying these numbers of PCs: {nComponentsList}\n")

# generate the fold splits grouped by patient
splits = groupedKFoldSplits(groups, nFolds=nFolds)

# this dictionary will store the PCA results for every fold and every nComponents
# structure: pcaResults[nComponents] = list of fold result dictionaries
pcaResults = {n: [] for n in nComponentsList}

# loop over each fold
for foldIndex, (trainIndices, testIndices) in enumerate(splits):

    # slice the data into training and test sets for this fold
    xTrain = X[trainIndices]
    xTest  = X[testIndices]
    yTrain = y[trainIndices]
    yTest  = y[testIndices]

    # print a summary of this fold so we can verify the split looks reasonable
    print(f"Fold {foldIndex + 1}:")
    print(f"  Train: {len(trainIndices)} samples "
          f"({np.sum(yTrain==1)} inflamed, {np.sum(yTrain==0)} non-inflamed)")
    print(f"  Test:  {len(testIndices)} samples "
          f"({np.sum(yTest==1)} inflamed, {np.sum(yTest==0)} non-inflamed)")

    # safety check: make sure no patient appears in both train and test
    trainPatients = set(groups[trainIndices])
    testPatients  = set(groups[testIndices])
    overlap       = trainPatients & testPatients  # intersection should be empty
    assert len(overlap) == 0, \
        f"Patient leakage detected in fold {foldIndex + 1}: {overlap}"

    # now run PCA for each number of components we want to try
    for nComponents in nComponentsList:

        # fit PCA using ONLY the training data
        # this gives us the components and mean from the training set
        components, mean, evRatio = pcaFit(xTrain, nComponents)

        # transform the training data into PCA space
        xTrainPca = pcaTransform(xTrain, components, mean)

        # transform the test data using the SAME components and mean from training
        # this is the correct way — test data is never used to fit PCA
        xTestPca = pcaTransform(xTest, components, mean)

        # store everything we need for kNN and XGBoost
        pcaResults[nComponents].append({
            "fold":      foldIndex,    # which fold this is
            "trainIdx":  trainIndices, # which samples were used for training
            "testIdx":   testIndices,  # which samples were used for testing
            "xTrainPca": xTrainPca,    # training features in PCA space
            "xTestPca":  xTestPca,     # test features in PCA space
            "yTrain":    yTrain,       # training labels
            "yTest":     yTest,        # test labels
            "evRatio":   evRatio,      # fraction of total variance each PC captures
        })

    print()  # blank line between folds for readability


# ── print variance explained summary ──────────────────────────────────────────

print("Variance explained (averaged across all folds):")

for nComponents in nComponentsList:

    # average the total variance explained across all 5 folds
    avgVarianceExplained = np.mean([
        foldData["evRatio"].sum()
        for foldData in pcaResults[nComponents]
    ])

    # this should now show realistic numbers like 35%, 50%, 65%
    # instead of 100% which was wrong
    print(f"  {nComponents} PCs: {avgVarianceExplained * 100:.1f}% of total variance explained")


# ── save results for kNN and XGBoost ──────────────────────────────────────────

# save the pcaResults dictionary so teammates can load it directly
# kNN and XGBoost will load this file and loop over folds
np.save(outputPath, pcaResults)

print(f"\nSaved PCA splits to {outputPath}")
print("\nHow to load and use this file:")
print("  pcaResults = np.load('Results/PCA/pca_splits.npy', allow_pickle=True).item()")
print("  pcaResults[10]                 -> list of 5 fold dicts for nComponents=10")
print("  pcaResults[10][0]              -> results for fold 0")
print("  pcaResults[10][0]['xTrainPca'] -> training features for fold 0")
print("  pcaResults[10][0]['xTestPca']  -> test features for fold 0")
print("  pcaResults[10][0]['yTrain']    -> training labels for fold 0")
print("  pcaResults[10][0]['yTest']     -> test labels for fold 0")