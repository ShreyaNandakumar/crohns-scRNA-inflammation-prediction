#import packages
import numpy as np
import xgboost as xgb
import os

#STEP 1: Load raw sample level data

#load raw features, lables, and patient groups

x_raw = np.load("Data/SampleLevelDataFinalUse/sample_level_X.npy")
y = np.load("Data/SampleLevelDataFinalUse/sample_level_y.npy")
groups = np.load("Data/SampleLevelDataFinalUse/sample_level_groups.npy", allow_pickle=True)

print("X_raw shape: " + str(x_raw.shape))   # should be (49, 2000)
print("y shape: "     + str(y.shape))        # should be (49,)
print("groups shape: "+ str(groups.shape))   # should be (49,)
print("Unique patients: " + str(len(np.unique(groups))))

#STEP 2: Create grouped CV splits manually 

#get unique patient groups
unique_patients = np.unique(groups)
n_patients = len(unique_patients)

#5 fold grouped CV - split patients into 5 groups
n_folds = 5
fold_size = n_patients // n_folds

#assign each patient to a fold
patient_to_fold = {}
for i, patient in enumerate(unique_patients):
    patient_to_fold[patient] = i // fold_size

#build fold splits
raw_splits = []
for fold in range(n_folds):

    #test patients are those assigned to the current fold
    test_patients = [patient for patient, assigned_fold in patient_to_fold.items() if assigned_fold == fold]
    train_patients = [patient for patient, assigned_fold in patient_to_fold.items() if assigned_fold != fold]

    #get indices for train and test samples based on patient groups
    test_idx = np.where(np.isin(groups, test_patients))[0]
    train_idx = np.where(np.isin(groups, train_patients))[0]

    raw_splits.append({
        "fold"   : fold,
        "xTrain" : x_raw[train_idx],
        "xTest"  : x_raw[test_idx],
        "yTrain" : y[train_idx],
        "yTest"  : y[test_idx]
    })

    print("Fold " + str(fold) +
          " train: " + str(len(train_idx)) +
          " test: "  + str(len(test_idx)))
    

#STEP 3: Define metrics from scratch (same as XGBoost.py)

def compute_MacroF1(y_true, y_pred):
    f1_scores = []
    for cls in [0, 1]:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def compute_AUROC(y_true, y_prob):
    sorted_indices = np.argsort(y_prob)[::-1]
    y_sorted_true  = y_true[sorted_indices]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tprs = [0.0]
    fprs = [0.0]
    tp   = 0
    fp   = 0

    for label in y_sorted_true:
        if label == 1:
            tp += 1
        else:
            fp += 1

        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)

    auroc = np.trapz(np.array(tprs), x=np.array(fprs))
    return float(auroc)

#STEP 4: Train XGBoost on raw features with both combos

#best AUROC combo from XGBoost.py
combo_best_auroc = {"n_components": 15,
                    "n_estimators": 100,
                    "max_depth"   : 3,
                    "learning_rate": 0.01,
                    "scale_pos_weight": 2.0}

#best balanced combo from XGBoost.py
combo_best_balanced = {"n_components": 15,
                    "n_estimators": 200,
                    "max_depth"   : 2,
                    "learning_rate": 0.3,
                    "scale_pos_weight": 2.0}

#make output directory
os.makedirs("Results/XGBoost_raw_vs_pca", exist_ok=True)

#loop over both the combos
for combo_name, combo in [("best_auroc", combo_best_auroc), ("best_balance", combo_best_balanced)]:

    print("Running combo: " + combo_name)

    fold_aurocs = []
    fold_F1s = []

    for fold_data in raw_splits:

        #get data for this fold
        xTrain = fold_data["xTrain"]
        yTrain = fold_data["yTrain"]
        xTest  = fold_data["xTest"]
        yTest  = fold_data["yTest"]

        #train XGBoost model with current combo params
        model = xgb.XGBClassifier(
            n_estimators=combo["n_estimators"],
            max_depth=combo["max_depth"],
            learning_rate=combo["learning_rate"],
            scale_pos_weight=combo["scale_pos_weight"],
            use_label_encoder=False,
            eval_metric="logloss"
        )

        model.fit(xTrain, yTrain)

        #predict probabilities and classes
        y_prob = model.predict_proba(xTest)[:, 1]
        y_pred = model.predict(xTest)

        #compute metrics
        auroc_score = compute_AUROC(yTest, y_prob)
        f1_score    = compute_MacroF1(yTest, y_pred)

        fold_aurocs.append(auroc_score)
        fold_F1s.append(f1_score)

    #average metrics across folds
    avg_F1 = float(np.nanmean(fold_F1s))
    std_F1 = float(np.nanstd(fold_F1s))
    valid_AUCs = [a for a in fold_aurocs if not np.isnan(a)]
    avg_AUROC = float(np.nanmean(valid_AUCs)) if valid_AUCs else float("nan")
    std_AUROC = float(np.nanstd(valid_AUCs)) if valid_AUCs else float("nan")

    print("  avg F1:    " + str(round(avg_F1,    4)) +
          " +/- "         + str(round(std_F1,    4)))
    print("  avg AUROC: " + str(round(avg_AUROC, 4)) +
            " +/- "         + str(round(std_AUROC, 4)))
    

#STEP 5: Print comparison table
print("\n")
print("=" * 60)
print("COMPARISON: Raw features vs PCA features (XGBoost)")
print("=" * 60)
print("{:<35} {:>8} {:>8}".format("Model", "F1", "AUROC"))
print("-" * 60)
print("{:<35} {:>8} {:>8}".format("XGBoost + PCA 15 (best AUROC)",    "0.6084", "0.8429"))
print("{:<35} {:>8} {:>8}".format("XGBoost + PCA 10 (best balanced)", "0.7212", "0.8324"))
print("{:<35} {:>8} {:>8}".format("XGBoost + Raw (best AUROC)",       "0.6879", "0.7935"))
print("{:<35} {:>8} {:>8}".format("XGBoost + Raw (best balanced)",    "0.7924", "0.8156"))