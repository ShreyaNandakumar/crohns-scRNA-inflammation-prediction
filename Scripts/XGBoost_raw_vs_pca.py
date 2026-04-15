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
