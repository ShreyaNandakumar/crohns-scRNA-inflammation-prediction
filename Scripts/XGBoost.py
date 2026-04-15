#import the packages
import numpy as np
import xgboost as xgb
import os

#STEP 1: Load the PCA splits created by SHreya (big)


#pca_splits.npy is a dictionary with keys as the split names and values as the PCA results for that split
#each key maps to a list of 5 totoal dictionaries
#each fold dictionary has: xTrainPCA, xTestPCA, yTrain, yTest, fold)
pca_results = np.load("Results/PCA/pca_splits.npy", allow_pickle=True).item()

print("PC options available: " + str(list(pca_results.keys())))
print("Number of folds: " + str(len(pca_results[10])))

#check the shape of the PCA results for nComponents = 10 
for fold_data in pca_results[10]:
    print("Fold " + str(fold_data['fold']) + "  xTrain: " + str(fold_data['xTrainPca'].shape) +
          "  xTest: "  + str(fold_data['xTestPca'].shape) +
          "  yTrain: " + str(fold_data['yTrain'].shape) +
          "  yTest: "  + str(fold_data['yTest'].shape)) 
    
"""
#STEP 2: Define the hyperparametrs and train XGBoost across folds

#sample ration: 28 inflammed/21 non-inflammed (1.33 approx.)
#scale_pos_weight = num_negative/num_positive to handle class imbalance
scale_pos_list = [1.0, 1.33, 2.0]

#hyparamters to try
n_est_list = [50, 100, 200]
max_depth_list = [2, 3, 4]
learning_rate_list = [0.01, 0.1, 0.3]
n_components_list = [5, 10, 15]

#store results
all_results = []

#loop over all the PC options
for n_components in n_components_list:
    print("n_components: " + str(n_components))

    #loop over the hyperparamter combinations
    for n_estimators in n_est_list:
        for max_depth in max_depth_list:
            for learning_rate in learning_rate_list:
                for scale_pos in scale_pos_list:

                    #store per-fold metrics
                    fold_F1s = []
                    fold_Aurocs = []

                    #loop over all 5 folds
                    for fold_data in pca_results[n_components]:

                        xTrain = fold_data['xTrainPca']
                        yTrain = fold_data['yTrain']
                        xTest = fold_data['xTestPca']
                        yTest = fold_data['yTest']
                        fold = fold_data['fold']

                        #define the XGBoost model
                        model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                                  learning_rate=learning_rate, 
                                                  scale_pos_weight=scale_pos, 
                                                  use_label_encoder=False, 
                                                  eval_metric='logloss', random_state=42)
                        
                        #train on the training fold
                        model.fit(xTrain, yTrain)

                        #predict the probabilities for AUROC
                        y_prob_a = model.predict_proba(xTest)[:, 1]

                        #predict class labels for F1
                        y_pred = model.predict(xTest)

                        #placeholders for now (metrics to be calculated in the next step)
                        fold_F1s.append(None)  #replace with actual F1 calculation
                        fold_Aurocs.append(None)  #replace with actual AUROC calculation

                        print("nComp=" + str(n_components) +
                          " nEst=" + str(n_estimators) +
                          " depth=" + str(max_depth) +
                          " lr=" + str(learning_rate) +
                          " scalePos=" + str(scale_pos) +
                          " training done chattttt")
                        
"""

#STEP 3: Calculate the F1 and AUROC metrics for each fold 

def compute_MacroF1(y_true, y_pred):
    #compute F1 seperately for each class and then take the average
    f1_scores = []
    for cls in [0, 1]: #binary classification
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
        f1_scores.append(f1)

    #macro = unweighted average across both classes
    return float(np.mean(f1_scores))

def compute_AUROC(y_true, y_prob):

    #sort samples by predicted probability (descending)
    sorted_indices = np.argsort(y_prob)[::-1]
    y_sorted_true = y_true[sorted_indices]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0) 

    #edge case: only 1 class present in the fold
    if n_pos == 0 or n_neg == 0:
        return float("nan") #AUROC undefined
    
    #build TPR and FPR at each threshold step
    tprs = [0.0]
    fprs = [0.0]
    tp = 0
    fp = 0

    for label in y_sorted_true:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)

    #area under ROC curve using trapezoidal rule
    auroc = np.trapz(np.array(tprs), x=np.array(fprs))
    
    return float(auroc)

scale_pos_list = [1.0, 1.33, 2.0]
n_est_list = [50, 100, 200]
max_depth_list = [2, 3, 4]
learning_rate_list = [0.01, 0.1, 0.3]
n_components_list = [5, 10, 15]

#STEP 4: Training loop - train XGBoost and compute metrics

#store all results (one row for 1 hyperparamter combo)
all_results = []

#make output directory if it doesn't exist
os.makedirs("Results/XGBoost", exist_ok=True)

#loop over all the PC options
for n_components in n_components_list:
    print("n_components: " + str(n_components))

    #loop over all hyperparamter combinations
    for n_estimators in n_est_list:
        for max_depth in max_depth_list:
            for learning_rate in learning_rate_list:
                for scale_pos in scale_pos_list:

                    #store per-fold metrics
                    fold_F1s = []
                    fold_Aurocs = []

                    #loop over all 5 folds
                    for fold_data in pca_results[n_components]:

                        xTrain = fold_data['xTrainPca']
                        yTrain = fold_data['yTrain']
                        xTest = fold_data['xTestPca']
                        yTest = fold_data['yTest']
                        fold = fold_data['fold']

                        #define and trainthe XGBoost model
                        model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                                  learning_rate=learning_rate, 
                                                  scale_pos_weight=scale_pos, 
                                                  use_label_encoder=False, 
                                                  eval_metric='logloss', random_state=42)
                        
                        model.fit(xTrain, yTrain)

                        #predicted probabilities for AUROC
                        y_prob_a = model.predict_proba(xTest)[:, 1]

                        """#predicted class labels for F1
                        y_pred = model.predict(xTest)

                        #compute metrics using scratch functions
                        f1 = compute_MacroF1(yTest, y_pred)
                        auroc = compute_AUROC(yTest, y_prob_a)"""

                        #threshold tuning using the TRAINING set
                        y_prob_train = model.predict_proba(xTrain)[:, 1]

                        thresholds = np.arange(0.1, 0.9, 0.01)
                        best_threshold = 0.5
                        best_f1_train = -1

                        for thresh in thresholds:
                            y_train_pred_thresh = (y_prob_train >= thresh).astype(int)
                            f1_train = compute_MacroF1(yTrain, y_train_pred_thresh)

                            if f1_train > best_f1_train:
                                best_f1_train = f1_train
                                best_threshold = thresh
                        
                        #apply the best threshold to the TEST set
                        y_pred = (y_prob_a >= best_threshold).astype(int)

                        #compute metrics using scratch functions
                        f1 = compute_MacroF1(yTest, y_pred)
                        auroc = compute_AUROC(yTest, y_prob_a)

                        fold_F1s.append(f1)
                        fold_Aurocs.append(auroc)

                    #average metrics across folds
                    avg_F1 = float(np.nanmean(fold_F1s)) #nanmean to handle any folds with undefined metrics
                    std_F1 = float(np.nanstd(fold_F1s))
                    valid_AUCs = [a for a in fold_Aurocs if not np.isnan(a)]
                    mean_AUROC = float(np.mean(valid_AUCs)) if valid_AUCs else float("nan")
                    std_AUROC = float(np.std(valid_AUCs)) if valid_AUCs else float("nan")

                    #store results for this hyperparam combo
                    all_results.append({
                        "n_components": n_components,
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "scale_pos_weight": scale_pos,
                        "avg_F1": round(avg_F1, 4),
                        "std_F1": round(std_F1, 4),
                        "avg_AUROC": round(mean_AUROC, 4) if not np.isnan(mean_AUROC) else "nan",
                        "std_AUROC": round(std_AUROC, 4) if not np.isnan(std_AUROC) else "nan"
                    })

                    print("n_comp= " + str(n_components) +
                          " n_est=" + str(n_estimators) +
                          " depth=" + str(max_depth) +
                          " lr=" + str(learning_rate) +
                          " scalePos=" + str(scale_pos) +
                          " avg_F1=" + str(round(avg_F1, 4)) +
                          " avg_AUROC=" + (str(round(mean_AUROC, 4)) if not np.isnan(mean_AUROC) else "nan"))
                    

#STEP 5: Save the results to a CSV file

#make output directory if it doesn't exist
os.makedirs("Results/XGBoost", exist_ok=True)

#write results manually to a CSV file
csv_path = "Results/XGBoost/xgboost_results.csv"
header = "n_components,n_estimators,max_depth,learning_rate,scale_pos_weight,avg_F1,std_F1,avg_AUROC,std_AUROC\n"

with open(csv_path, "w") as f:
    f.write(header)
    for row in all_results:
        line = (str(row["n_components"]) + "," +
                str(row["n_estimators"]) + "," +
                str(row["max_depth"]) + "," +
                str(row["learning_rate"]) + "," +
                str(row["scale_pos_weight"]) + "," +
                str(row["avg_F1"]) + "," +
                str(row["std_F1"]) + "," +
                str(row["avg_AUROC"]) + "," +
                str(row["std_AUROC"]))
        
        f.write(line + "\n")

print("\nAll results saved to: " + csv_path)

#print top 5 combos sorted by mean AUROC
print("\nTop 5 hyperparameter combos by mean AUROC:")
sorted_results = sorted(all_results, key=lambda x: x["avg_AUROC"] if x["avg_AUROC"] != "nan" else -1, reverse=True)

for r in sorted_results[:5]:
    print("  nComp=" + str(r['n_components']) +
          " nEst="   + str(r['n_estimators']) +
          " depth="  + str(r['max_depth'])    +
          " lr="     + str(r['learning_rate'])+
          " scale="  + str(r['scale_pos_weight'])    +
          " F1="   + str(r['avg_F1'])      +
          " AUROC="+ str(r['avg_AUROC']))
    

#STEP 6: Save feature importances for best combo per fold

auroc_folder = "Results/XGBoost/AUROC_run"
balanced_folder = "Results/XGBoost/BALANCED_run"

#combo 1: highest AUROC - best at ranking but poor classification
best_auroc = sorted_results[0]

#combo 2: highest F1 - best at classification but poor ranking
#sort by F1 to find best balanced combo
sorted_by_f1 = sorted(all_results, key=lambda x: x["avg_F1"], reverse=True)
best_balanced = sorted_by_f1[0]

print("\nCombo 1 (highest AUROC): " +
      "nComp=" + str(best_auroc["n_components"]) +
      " nEst=" + str(best_auroc["n_estimators"]) +
      " depth=" + str(best_auroc["max_depth"]) +
      " lr=" + str(best_auroc["learning_rate"]) +
      " scale=" + str(best_auroc["scale_pos_weight"]))

print("Combo 2 (highest F1): " +
      "nComp=" + str(best_balanced["n_components"]) +
      " nEst=" + str(best_balanced["n_estimators"]) +
      " depth=" + str(best_balanced["max_depth"]) +
      " lr=" + str(best_balanced["learning_rate"]) +
      " scale=" + str(best_balanced["scale_pos_weight"]))

#loop over both the combos
for combo_name, combo in [("best_auroc", best_auroc), ("best_balanced", best_balanced)]:
    print("\nSaving feature importances for combo: " + combo_name)

    #retrain the model on each fold with this combo
    for fold_data in pca_results[combo["n_components"]]:
        xTrain = fold_data['xTrainPca']
        yTrain = fold_data['yTrain']
        fold = fold_data['fold']

        #train model
        model = xgb.XGBClassifier(n_estimators=combo["n_estimators"], 
                                  max_depth=combo["max_depth"], 
                                  learning_rate=combo["learning_rate"], 
                                  scale_pos_weight=combo["scale_pos_weight"], 
                                  use_label_encoder=False, 
                                  eval_metric='logloss', random_state=42)
        
        model.fit(xTrain, yTrain)

        #get feature importances
        importances = model.feature_importances_

        #save to csv file
        if combo_name == "best_auroc":
            folder = auroc_folder
        else:
            folder = balanced_folder

        imp_path = f"{folder}/feature_importance_{combo_name}_fold{fold}.csv"

        with open(imp_path, "w") as f:
            f.write("pc,importance\n")
            for i, imp in enumerate(importances):
                f.write("PC" + str(i + 1) + "," + str(round(imp, 6)) + "\n")

        print("  Fold " + str(fold) + " saved to " + imp_path)

print("\nALL feature importances saved woohoooo!")