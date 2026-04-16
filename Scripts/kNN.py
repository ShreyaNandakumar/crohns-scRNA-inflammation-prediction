import numpy as np 
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os 

pcaSplitsPath = "Results/PCA/pca_splits.npy"

outputDir = "Results/kNN"

pcaResults = np.load(pcaSplitsPath, allow_pickle=True).item()

def euclideanDistances(xTest, xTrain):
    """
    Calculate pairwise euclidean distances between every test and training sample.

    Parameters:
        xTest  : test feature matrix   (nTest  x nFeatures)
        xTrain : train feature matrix  (nTrain x nFeatures)

    Returns distances : matrix of distances (nTest x nTrain)
                    distances[i][j] = Euclidean distance from test sample i to training sample j
    """

    testSqNorms = np.sum(xTest ** 2, axis = 1, keepdims = True)
    trainSqNorms = np.sum(xTrain ** 2, axis = 1, keepdims = True).T
    crossTerm = xTest @ xTrain.T

    sqDistances = testSqNorms + trainSqNorms - 2 * crossTerm 
    sqDistances = np.maximum(sqDistances, 0.0)
    distances = np.sqrt(sqDistances)

    return distances 

def knnPredict(xTrain, yTrain, xTest, k):
    """
    Predict both labels and probabilities using k-Nearest Neighbors.

    For each test sample:
      1. compute Euclidean distance to every training sample
      2. find the k training samples with the smallest distances
      3. compute probability as proportion of k neighbors in class 1
      4. majority vote for the hard label (ties broken by nearest neighbor)

    Parameters: xTrain, yTrain, xTest, k

    Returns: predictions and probabilities
    """

    # computing euclidean distance from every test point to every training point 
    distances = euclideanDistances(xTest, xTrain)

    #for each test sample, getting the indices of the k nearest training samples, sorting ascending order and then tae the first k value
    neighborIndices = np.argsort(distances, axis = 1,)[:, :k]

    #look up the labels of those k neighbors
    neighborLabels = yTrain[neighborIndices]

    #compute class-1 probabilities 
    probabilities = np.mean(neighborLabels == 1, axis = 1).astype(float)

    #hard label predictions 
    predictions = np.zeros(xTest.shape[0], dtype = int)

    for i in range(xTest.shape[0]):
        labels = neighborLabels[i]

        countClass0 = np.sum(labels == 0)
        countClass1 = np.sum(labels == 1)

        if countClass1 > countClass0:
            predictions[i] = 1
        elif countClass0 > countClass1:
            predictions[i] = 0
        else: 
            nearestIdx = neighborIndices[i,0]
            predictions[i] = yTrain[nearestIdx]

    return predictions, probabilities

def accuracy(yTrue, yPred):
    """
    Compute classification accuracy.

    accuracy = number of correct predictions / total predictions

    Parameters: yTrue, yPred

    Returns: accuracy as a float between 0 and 1
    """

    correct = np.sum(yTrue == yPred)
    total = len(yTrue)

    return correct / total


def confusionmatrix(yTrue, yPred):
    """
    Compute a 2x2 confusion matrix for binary classification.

    Where:
        TN = true negatives  (predicted 0, actual 0)
        FP = false positives (predicted 1, actual 0)
        FN = false negatives (predicted 0, actual 1)
        TP = true positives  (predicted 1, actual 1)

    Parameters: yTrue, yPred 

    Returns: confusion matrix 
    """

    tn = np.sum((yTrue == 0) & (yPred == 0))
    fp = np.sum((yTrue == 0) & (yPred == 1))
    fn = np.sum((yTrue == 1) & (yPred == 0))
    tp = np.sum((yTrue == 1) & (yPred == 1))

    cm = np.array([[tn, fp],
                    [fn, tp]])
    
    return cm 

def F1_score(yTrue, yPred):
    """
    Compute the macro-averaged F1 score for binary classification.

    Macro F1 computes F1 for each class separately, then averages them.
    This treats both classes equally regardless of their frequency,
    which is important since our dataset is slightly imbalanced
    (28 inflamed vs 21 non-inflamed).

    F1 for a single class = 2 * precision * recall / (precision + recall)
    macro F1 = (F1_class0 + F1_class1) / 2

    Parameters: yTrue, yPred 

    Returns: macro F1 score as a float between 0 and 1
    """
    f1Scored = []

    for cls in [0, 1]:
        truePositives = np.sum((yTrue == cls) & (yPred == cls))
        predictedPositives = np.sum(yPred == cls)

        if predictedPositives == 0:
            precision = 0.0
        else: 
            precision = truePositives / predictedPositives
        
        actualPositives = np.sum(yTrue == cls)

        if actualPositives == 0:
            recall = 0.0
        else: 
            recall = truePositives / actualPositives

        if precision + recall == 0:
            f1 = 0.0
        else: 
            f1 = 2 * precision * recall / (precision + recall)
        
        f1Scored.append(f1)

    return np.mean(f1Scored)

def ROCurve(yTrue, yScores):
    """
    Compute the ROC curve (FPR, TPR) and AUROC from scratch.

    Parameters: yTrue  , yScores 

    Returns: fpr, tpr, auroc 
    """
    thresholds = np.unique(yScores)
    thresholds = np.sort(thresholds)[::-1]

    thresholds = np.concatenate([[thresholds[0] + 1.0], thresholds])

    totalPositives = np.sum(yTrue == 1)
    totalNegatives = np.sum(yTrue == 0)

    fprList = []
    tprList = []

    for thresh in thresholds:
        yPred = (yScores >= thresh).astype(int)

        tp = np.sum((yPred == 1) & (yTrue == 1))
        fp = np.sum((yPred == 1) & (yTrue == 0))

        tpr = tp / totalPositives if totalPositives > 0 else 0.0
        fpr = fp / totalNegatives if totalNegatives > 0 else 0.0

        tprList.append(tpr)
        fprList.append(fpr)
    
    fpr = np.array(fprList)
    tpr = np.array(tprList)

    auroc = np.trapezoid(tpr, fpr)

    return fpr, tpr, auroc


# running knn inside cross validation 

nComponentsList = sorted(pcaResults.keys())
kValues = [1,3,5,7]
nFolds = len(pcaResults[nComponentsList[0]])

print(f"\nRunning kNN with grouped cross-validation")
print(f"nComponents to try: {nComponentsList}")
print(f"k values to try:    {kValues}")
print(f"Number of folds:    {nFolds}\n")

allResults = {}

for nComponents in nComponentsList:
    for k in kValues:
        foldAccuracies = []
        foldF1s = []
        foldCMs = []
        foldAurocs = []
        foldFprs = []
        foldTprs = []
        for foldIndex in range(nFolds):
            foldData = pcaResults[nComponents][foldIndex]

            xTrainPca = foldData["xTrainPca"]
            xTestPca  = foldData["xTestPca"]
            yTrain    = foldData["yTrain"]
            yTest     = foldData["yTest"]

            yPred, yProba = knnPredict(xTrainPca, yTrain, xTestPca, k)

            foldAcc = accuracy(yTest, yPred)
            foldF1  = F1_score(yTest, yPred)
            foldCM  = confusionmatrix(yTest, yPred)
            fpr, tpr, auroc = ROCurve(yTest, yProba)

            foldAccuracies.append(foldAcc)
            foldF1s.append(foldF1)
            foldCMs.append(foldCM)
            foldAurocs.append(auroc)
            foldFprs.append(fpr)
            foldTprs.append(tpr)

        meanAcc = np.mean(foldAccuracies)
        stdAcc  = np.std(foldAccuracies)
        meanF1  = np.mean(foldF1s)
        stdF1   = np.std(foldF1s)
        meanAuroc = np.mean(foldAurocs)
        stdAuroc  = np.std(foldAurocs)

        aggregateCM = np.sum(foldCMs, axis=0)

        allResults[(nComponents, k)] = {
            "foldAccuracies": foldAccuracies,
            "foldF1s":        foldF1s,
            "foldCMs":        foldCMs,
            "foldAurocs":     foldAurocs,
            "foldFprs":       foldFprs,
            "foldTprs":       foldTprs,
            "meanAccuracy":   meanAcc,
            "stdAccuracy":    stdAcc,
            "meanF1":         meanF1,
            "stdF1":          stdF1,
            "meanAuroc":      meanAuroc,
            "stdAuroc":       stdAuroc,
            "aggregateCM":    aggregateCM,
        }

# summary table
print("SUMMARY: kNN Results (mean ± std across 5 folds)")
print(f"{'nPCs':<8} {'k':<6} {'Accuracy':<20} {'Macro F1':<20} {'AUROC':<20}")
print("-" * 88)

bestKey = None
bestF1  = -1.0

for nComponents in nComponentsList:
    for k in kValues:
        res = allResults[(nComponents, k)]

        accStr   = f"{res['meanAccuracy']:.3f} ± {res['stdAccuracy']:.3f}"
        f1Str    = f"{res['meanF1']:.3f} ± {res['stdF1']:.3f}"
        aurocStr = f"{res['meanAuroc']:.3f} ± {res['stdAuroc']:.3f}"

        print(f"{nComponents:<8} {k:<6} {accStr:<20} {f1Str:<20} {aurocStr:<20}")

        # track the best (nComponents, k) by macro F1
        if res["meanF1"] > bestF1:
            bestF1  = res["meanF1"]
            bestKey = (nComponents, k)

print("-" * 88)
print(f"\nBest setting: nComponents={bestKey[0]}, k={bestKey[1]} "
    f"(macro F1 = {bestF1:.3f}, AUROC = {allResults[bestKey]['meanAuroc']:.3f})")


bestResult = allResults[bestKey]


# save summary CSV (one row per hyperparameter combination) - main results

summaryRows = []

for nComponents in nComponentsList:
    for k in kValues:
        res = allResults[(nComponents, k)]
        aggCM = res["aggregateCM"]

        summaryRows.append([
            nComponents,
            k,
            round(res["meanAccuracy"], 4),
            round(res["stdAccuracy"], 4),
            round(res["meanF1"], 4),
            round(res["stdF1"], 4),
            round(res["meanAuroc"], 4),
            round(res["stdAuroc"], 4),
            int(aggCM[1, 1]),  # TP
            int(aggCM[0, 0]),  # TN
            int(aggCM[0, 1]),  # FP
            int(aggCM[1, 0]),  # FN
        ])

# write CSV manually (no pandas dependency needed)
summaryPath = os.path.join(outputDir, "knn_results_summary.csv")
header = "nComponents,k,meanAccuracy,stdAccuracy,meanMacroF1,stdMacroF1,meanAUROC,stdAUROC,TP,TN,FP,FN"

with open(summaryPath, "w") as f:
    f.write(header + "\n")
    for row in summaryRows:
        f.write(",".join(str(v) for v in row) + "\n")

print(f"\nSaved summary CSV to {summaryPath}")

# save per-fold CSV (one row per fold per hyperparameter combination) - this has the detailed fold-by-fold breakdown

foldRows = []

for nComponents in nComponentsList:
    for k in kValues:
        res = allResults[(nComponents, k)]
        for foldIndex in range(nFolds):
            cm = res["foldCMs"][foldIndex]
            foldRows.append([
                nComponents,
                k,
                foldIndex + 1,
                round(res["foldAccuracies"][foldIndex], 4),
                round(res["foldF1s"][foldIndex], 4),
                round(res["foldAurocs"][foldIndex], 4),
                int(cm[1, 1]),  # TP
                int(cm[0, 0]),  # TN
                int(cm[0, 1]),  # FP
                int(cm[1, 0]),  # FN
            ])

foldPath = os.path.join(outputDir, "knn_results_per_fold.csv")
foldHeader = "nComponents,k,fold,accuracy,macroF1,AUROC,TP,TN,FP,FN"

with open(foldPath, "w") as f:
    f.write(foldHeader + "\n")
    for row in foldRows:
        f.write(",".join(str(v) for v in row) + "\n")

print(f"Saved per-fold CSV to {foldPath}")

#plots

#Aggregate confusion matrix heatmap for best setting 

def plotConfusionMatrix(cm, title, savePath):
    """
    Plot a 2x2 confusion matrix as a colour-coded heatmap.

    Parameters:
        cm       : 2x2 numpy array (confusion matrix)
        title    : plot title string
        savePath : where to save the .png file
    """

    fig, ax = plt.subplots(figsize=(6, 5))

    # show the matrix as a heatmap with blue colourmap
    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax, ax=ax, shrink=0.8)

    # annotate each cell with its count
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            # use white text on dark cells, black on light cells
            cellValue = cm[row, col]
            maxVal = cm.max()
            textColour = "white" if cellValue > maxVal / 2 else "black"
            ax.text(col, row, str(int(cellValue)),
                    ha="center", va="center", fontsize=20, fontweight="bold",
                    color=textColour)

    # axis labels
    classNames = ["Non-Inflamed", "Inflamed"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classNames, fontsize=12)
    ax.set_yticklabels(classNames, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("Actual Label", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(savePath, dpi=300)
    plt.close()
    print(f"  Saved: {savePath}")


# aggregate confusion matrix for best hyperparameter setting
plotConfusionMatrix(
    bestResult["aggregateCM"],
    f"kNN Confusion Matrix (Aggregate)\nnPCs={bestKey[0]}, k={bestKey[1]}",
    os.path.join(outputDir, "knn_confusion_matrix_aggregate.png")
)

# per-fold confusion matrices in a single figure
fig, axes = plt.subplots(1, nFolds, figsize=(4 * nFolds, 4))

for foldIndex in range(nFolds):
    ax = axes[foldIndex]
    cm = bestResult["foldCMs"][foldIndex]

    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            cellValue = cm[row, col]
            maxVal = max(cm.max(), 1)  # avoid division by zero for empty folds
            textColour = "white" if cellValue > maxVal / 2 else "black"
            ax.text(col, row, str(int(cellValue)),
                    ha="center", va="center", fontsize=16, fontweight="bold",
                    color=textColour)

    classNames = ["Non-Inf", "Inflamed"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classNames, fontsize=9)
    ax.set_yticklabels(classNames, fontsize=9)
    ax.set_title(f"Fold {foldIndex + 1}", fontsize=12, fontweight="bold")

    if foldIndex == 0:
        ax.set_ylabel("Actual", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)

fig.suptitle(f"kNN Per-Fold Confusion Matrices (nPCs={bestKey[0]}, k={bestKey[1]})",
            fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(outputDir, "knn_confusion_matrix_per_fold.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {os.path.join(outputDir, 'knn_confusion_matrix_per_fold.png')}")


# Accuracy and Macro F1 grouped bar chart

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# prepare data for the bar chart
# x-axis: each (nPCs, k) combination
barLabels = []
accMeans  = []
accStds   = []
f1Means   = []
f1Stds    = []

for nComponents in nComponentsList:
    for k in kValues:
        barLabels.append(f"PCs={nComponents}\nk={k}")
        res = allResults[(nComponents, k)]
        accMeans.append(res["meanAccuracy"])
        accStds.append(res["stdAccuracy"])
        f1Means.append(res["meanF1"])
        f1Stds.append(res["stdF1"])

xPos = np.arange(len(barLabels))
barWidth = 0.65

# define colours: one colour per nComponents group
colours = []
colourMap = {5: "#4C72B0", 10: "#DD8452", 15: "#55A868"}
for nComponents in nComponentsList:
    for k in kValues:
        colours.append(colourMap[nComponents])

# accuracy bars
ax1 = axes[0]
bars1 = ax1.bar(xPos, accMeans, barWidth, yerr=accStds,
                color=colours, edgecolor="black", linewidth=0.5,
                capsize=4, alpha=0.85)
ax1.set_ylabel("Accuracy", fontsize=13)
ax1.set_title("kNN Accuracy Across Hyperparameters", fontsize=14, fontweight="bold")
ax1.set_xticks(xPos)
ax1.set_xticklabels(barLabels, fontsize=8)
ax1.set_ylim(0, 1.0)
ax1.axhline(y=28/49, color="gray", linestyle="--", linewidth=1, label="Majority baseline")
ax1.legend(fontsize=10)

# macro F1 bars
ax2 = axes[1]
bars2 = ax2.bar(xPos, f1Means, barWidth, yerr=f1Stds,
                color=colours, edgecolor="black", linewidth=0.5,
                capsize=4, alpha=0.85)
ax2.set_ylabel("Macro F1", fontsize=13)
ax2.set_title("kNN Macro F1 Across Hyperparameters", fontsize=14, fontweight="bold")
ax2.set_xticks(xPos)
ax2.set_xticklabels(barLabels, fontsize=8)
ax2.set_ylim(0, 1.0)

# add a legend for nComponents groups
from matplotlib.patches import Patch
legendHandles = [Patch(facecolor=colourMap[n], edgecolor="black", label=f"nPCs={n}")
                for n in nComponentsList]
ax2.legend(handles=legendHandles, fontsize=10)

plt.tight_layout()
barChartPath = os.path.join(outputDir, "knn_accuracy_f1_comparison.png")
plt.savefig(barChartPath, dpi=300)
plt.close()
print(f"  Saved: {barChartPath}")


# Macro F1 line plot (k on x-axis, one line per nComponents)

fig, ax = plt.subplots(figsize=(8, 5))

markers = ["o", "s", "D"]
lineColours = ["#4C72B0", "#DD8452", "#55A868"]

for idx, nComponents in enumerate(nComponentsList):
    f1Vals   = [allResults[(nComponents, k)]["meanF1"] for k in kValues]
    f1Errors = [allResults[(nComponents, k)]["stdF1"]  for k in kValues]

    ax.errorbar(kValues, f1Vals, yerr=f1Errors,
                marker=markers[idx], color=lineColours[idx],
                linewidth=2, markersize=8, capsize=5,
                label=f"nPCs={nComponents}")

ax.set_xlabel("k (number of neighbors)", fontsize=13)
ax.set_ylabel("Macro F1 (mean ± std)", fontsize=13)
ax.set_title("kNN Macro F1 vs. k by Number of PCs", fontsize=14, fontweight="bold")
ax.set_xticks(kValues)
ax.legend(fontsize=11)
ax.set_ylim(0.3, 0.85)
ax.grid(True, alpha=0.3)

plt.tight_layout()
linePlotPath = os.path.join(outputDir, "knn_f1_vs_k.png")
plt.savefig(linePlotPath, dpi=300)
plt.close()
print(f"  Saved: {linePlotPath}")


#Plot 4: ROC Curve for best hyperparameter setting 

fig, ax = plt.subplots(figsize=(8, 7))

# plot per-fold ROC curves
meanFprInterp = np.linspace(0, 1, 200)
interpTprs = []

for foldIndex in range(nFolds):
    fpr = bestResult["foldFprs"][foldIndex]
    tpr = bestResult["foldTprs"][foldIndex]
    auroc = bestResult["foldAurocs"][foldIndex]

    ax.plot(fpr, tpr, alpha=0.3, linewidth=1.2,
            label=f"Fold {foldIndex + 1} (AUROC = {auroc:.3f})")

    # interpolate TPR onto a common FPR grid for computing mean curve
    interpTpr = np.interp(meanFprInterp, fpr, tpr)
    interpTpr[0] = 0.0  # ensure curve starts at (0, 0)
    interpTprs.append(interpTpr)

# compute and plot the mean ROC curve
meanTpr = np.mean(interpTprs, axis=0)
meanTpr[-1] = 1.0  # ensure curve ends at (1, 1)
meanAuroc = bestResult["meanAuroc"]
stdAuroc  = bestResult["stdAuroc"]

ax.plot(meanFprInterp, meanTpr, color="blue", linewidth=2.5,
        label=f"Mean ROC (AUROC = {meanAuroc:.3f} ± {stdAuroc:.3f})")

# shade ±1 std dev region
stdTpr = np.std(interpTprs, axis=0)
tprUpper = np.minimum(meanTpr + stdTpr, 1)
tprLower = np.maximum(meanTpr - stdTpr, 0)
ax.fill_between(meanFprInterp, tprLower, tprUpper, color="blue", alpha=0.1,
                label="± 1 std. dev.")

# diagonal (random classifier)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1,
        label="Random (AUROC = 0.500)")

ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title(f"kNN ROC Curve (nPCs={bestKey[0]}, k={bestKey[1]})",
            fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)

plt.tight_layout()
rocPath = os.path.join(outputDir, "knn_roc_curve.png")
plt.savefig(rocPath, dpi=300)
plt.close()
print(f"  Saved: {rocPath}")
